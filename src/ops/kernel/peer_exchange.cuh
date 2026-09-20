// ===========================================================================
// PROVENANCE: ported from github.com/ivanov84/ninfer-windows-tp2
//             (branch windows-tp2). Upstream lineage: NInfer TP2.
//             See LICENSE_ref for the license terms.
//             Ported on 2026-09-20 to give this tree the pinned-host peer
//             mailbox transport for Windows TP2 collectives.
//             Content kept verbatim apart from this banner.
// ===========================================================================
#pragma once

// ninfer::ops::detail - the kernel behind the pinned-host mailbox transport for TP2 collectives.
//
// TRANSPORT CONTRACT (why this exists). On this machine (Windows 11 WDDM, 2x RTX 5060 Ti,
// cudaDeviceCanAccessPeer == 0) a cross-device cudaMemcpyAsync is transparently staged by the
// driver through host memory by way of the copy engine, and the event chain that orders the
// staged copy costs far more than the payload: the measured production allreduce is ~277 us per
// 10 KiB reduction (128 per decode token = ~35 ms of a ~61 ms token), of which useful transfer
// time is ~3.3 us. This kernel pair replaces the whole choreography: both ranks run one kernel
// CONCURRENTLY, publish their operand into their own pinned host slot, release a flag, spin on
// the peer's flag, then sum locally -- no events, no copy engine, no driver round trip inside
// the exchange. Measured on this machine: ~41 us per 10 KiB reduction (graph-replayed), 6.7x
// faster than the staged path.
//
// ORDERING PROTOCOL (per slot; every captured collective call site owns one).
//
//   publish:  every thread stores its 16-byte payload chunks to the rank's host slot
//             __threadfence_system()          -- payload stores leave this GPU for system memory
//             __syncthreads()                 -- the block meets
//             one thread per block: atomicAdd(arrival, 1)
//             the block that observes arrival == gridDim.x - 1 (all blocks fenced and arrived):
//                 *flag = 1                   -- release: publish is globally complete
//                 *arrival = 0                -- restore for the next replay of this slot
//   consume:  one thread per block spins on the peer flag (volatile: system-memory reads are
//             uncached on the GPU side, so the poll always observes RAM)
//             __threadfence_system()          -- acquire: payload reads ordered after the flag
//             __syncthreads()
//             every thread reads its peer payload chunks and combines
//
// Both ranks execute the same schedule, so the exchange completes as soon as each side has
// observed the other's release -- a lock-step pipeline, no circular waits: rank A's kernel for
// slot k depends only on rank B's PUBLISH for slot k, which runs on the other GPU.
//
// FLAG LIFECYCLE. Flags are host words zeroed by the engine between graph replays (the round is
// fully retired before the next launch, so the reset cannot race any GPU access). Within one
// replay each slot fires exactly once (it belongs to one captured call site), so a poller can
// never observe a stale nonzero flag from an earlier call in the same replay. Payload slots are
// per-call-site too, so a publish can never overwrite a payload a lagging consumer still reads.
//
// HANG GUARD. A poller gives up after kPeerSpinLimit probes and sets `hang`, so a broken peer
// surfaces as a reported fault instead of a silent spin until the WDDM watchdog resets the
// device. The engine checks the aggregate hang word once per round when the egress is read.
//
// ARITHMETIC. The combine is the qualified residual_add body: FP32 accumulation of the two
// represented BF16 operands, one round-to-nearest-even on store. This matches the staged path's
// local combine bit for bit (the same two partials, the same order, the same rounding), so the
// mailbox transport cannot change a single output value.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// ~0.4 s of __nanosleep(100) polling before the poller gives up and reports: deliberately well
// under the ~2 s Windows WDDM watchdog timeout for a GPU hang, so a broken peer surfaces as a
// reported fault (the engine checks the aggregate word once per round) instead of a TDR device
// reset that takes the driver down with it.
inline constexpr std::uint32_t kPeerSpinLimit = 4000000u;

using PeerVec = uint4;  // 16 bytes = 8 BF16 elements, one PCIe transaction per access

// 16-byte chunks per thread per pass. A warp then touches 4 consecutive 512 B lines of the
// payload: 64 B per thread across a 2 KB span, so both the host write burst and the read phase
// coalesce into 128 B PCIe transactions instead of per-16 B ones.
inline constexpr int kPeerGroup = 4;

union PeerVecBf16 {
    PeerVec raw;
    __nv_bfloat162 pair[4];
};

// One rank's half of the two-rank allreduce exchange.
//
//   local_partial / out_sum      operand and in-place sum, both in this device's VRAM
//   mine_payload / mine_flag      this rank's pinned host publish slot and release flag
//   peer_payload / peer_flag      the peer's
//   arrival                       this slot's block-arrival counter, in this device's VRAM
//   hang                          aggregate hang-guard word, in this device's VRAM: a fault is
//                                 reported by writing VRAM (no PCIe) and the engine copies the
//                                 word out once per round with the egress. Keeping it OUT of the
//                                 pinned slab removes a system-memory broadcast read (one PCIe
//                                 TLP per warp) from every thread's exit path -- on the 40 KiB
//                                 MTP-verify width that is 48 extra link transactions per rank.
//   vecs                          payload length in 16-byte units
//
// Launch geometry: any grid, 256 threads. For the 10 KiB decode activation (640 vectors) the
// launcher picks 1 block; wider payloads scale to more blocks so the read phase can overlap
// PCIe latency across SMs instead of serializing behind one block's load pipeline.
__global__ __launch_bounds__(256) void peer_exchange_sum_kernel(
    const PeerVecBf16* __restrict__ local_partial, PeerVecBf16* __restrict__ out_sum,
    PeerVecBf16* __restrict__ mine_payload, volatile std::uint32_t* __restrict__ mine_flag,
    const PeerVecBf16* __restrict__ peer_payload, volatile std::uint32_t* __restrict__ peer_flag,
    std::uint32_t* __restrict__ arrival, volatile std::uint32_t* __restrict__ hang,
    int vecs) {
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int span = gridDim.x * blockDim.x;

    // Publish this rank's operand into its pinned host slot. Copies go through the trivial
    // `raw` member: the union's BF16 members have non-trivial special members, which would make
    // the whole union non-copyable in device code. Four 16-byte chunks per thread per pass so a
    // warp issues one 2 KB contiguous store burst (16 x 128 B PCIe write TLPs) instead of four
    // 512 B ones -- the wider the posted-write burst, the fewer chipset turns the flush costs.
    const int group_span = span * kPeerGroup;
    for (int g = tid * kPeerGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < kPeerGroup; ++j) {
            if (g + j < vecs) { mine_payload[g + j].raw = local_partial[g + j].raw; }
        }
    }
    __threadfence_system();
    __syncthreads();

    if (threadIdx.x == 0) {
        // Every block fences before it arrives, so the last arrival implies every payload
        // store from every block is already system-visible.
        const std::uint32_t arrived = atomicAdd(arrival, 1u);
        if (arrived == gridDim.x - 1u) {
            *mine_flag = 1;   // release
            *arrival   = 0;   // restore the counter for this slot's next replay
        }
        // Consume-side spin: system-memory reads are uncached, so this always observes RAM.
        __threadfence_system();
        std::uint32_t spins = 0;
        while (*peer_flag != 1u) {
            __nanosleep(100);
            if (++spins > kPeerSpinLimit) {
                *hang = 1;  // report the fault, then fall through: the block must still meet
                break;
            }
        }
        // Acquire: the peer payload reads below are ordered after the observed release.
        __threadfence_system();
    }
    __syncthreads();

    // Hang verdict: one system-memory read per BLOCK (thread 0), broadcast through shared -- not
    // one per warp, which on the 40 KiB width costs an extra PCIe TLP per warp on the exit path.
    __shared__ std::uint32_t hang_word;
    if (threadIdx.x == 0) { hang_word = *hang; }
    __syncthreads();
    if (hang_word != 0) { return; }

    // Combine in place: FP32 accumulate of the two represented BF16 operands, single
    // round-to-nearest-even on store. Grouped like the publish so the peer payload read is a
    // wide coalesced burst.
    for (int g = tid * kPeerGroup; g < vecs; g += group_span) {
#pragma unroll
        for (int j = 0; j < kPeerGroup; ++j) {
            if (g + j >= vecs) { break; }
            PeerVecBf16 mine;
            PeerVecBf16 peer;
            mine.raw = local_partial[g + j].raw;
            peer.raw = peer_payload[g + j].raw;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float a0 = __low2float(mine.pair[pair]);
                const float b0 = __high2float(mine.pair[pair]);
                const float a1 = __low2float(peer.pair[pair]);
                const float b1 = __high2float(peer.pair[pair]);
                mine.pair[pair] = __floats2bfloat162_rn(a0 + a1, b0 + b1);
            }
            out_sum[g + j].raw = mine.raw;
        }
    }
}

// Launch geometry for a payload of `bytes`: enough 16-byte vectors per thread per grouped pass to
// keep the consume phase latency-tolerant, capped so a 640-vector (10 KiB) reduction still runs
// as one block -- the single-block case avoids the arrival counter entirely on the common path.
inline int peer_exchange_blocks(int bytes) {
    const int vecs  = bytes / static_cast<int>(sizeof(PeerVec));
    const int want  = (vecs + kPeerGroup * 256 - 1) / (kPeerGroup * 256);
    return want < 1 ? 1 : (want > 16 ? 16 : want);
}

} // namespace ninfer::ops::detail
