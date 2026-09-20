// ===========================================================================
// PROVENANCE: ported from github.com/ivanov84/ninfer-windows-tp2
//             (branch windows-tp2). Upstream lineage: NInfer TP2.
//             See LICENSE_ref for the license terms.
//             Ported on 2026-09-20 to give this tree the pinned-host peer
//             mailbox transport for Windows TP2 collectives.
//             Content kept verbatim apart from this banner.
// ===========================================================================
#pragma once

// ninfer::ops - pinned-host mailbox transport for the two-device TP2 collectives.
//
// A PeerMailbox owns the pinned host memory the mailbox transport (see
// src/ops/kernel/peer_exchange.cuh) exchanges through, and installs itself as the process-wide
// transport for CAPTURED collectives. The staged, event-ordered path in allreduce.cu stays the
// only path for eager execution and oversized payloads; the mailbox serves the one shape that
// dominates decode -- small reductions replayed inside a captured CUDA graph -- where the
// measured cost is the choreography, not the bytes.
//
// MEASURED ON THIS MACHINE (2x RTX 5060 Ti, Windows 11 WDDM, no P2P):
//   staged event path (production allreduce_sum): ~277 us per 10 KiB reduction
//   mailbox exchange (graph replay):              ~41 us per 10 KiB reduction
// Over 128 reductions per decode token that is ~35 ms -> ~5 ms, the difference between a
// communication-bound decode and a compute-bound one.
//
// LIFECYCLE.
//   PeerMailbox mailbox(ec);                  once, at Program setup, next to PeerEvents
//                                             (cudaHostAlloc is not stream-ordered and must
//                                             never appear in a hot path or a capture)
//   ... capture the decode program ...
//   PeerMailbox::reset_host_flags();          once per round, between graph replays, on the
//                                             host, after the round retired on both devices
//
// WHY A PROCESS-WIDE INSTALL RATHER THAN A PARAMETER. The collective is issued deep inside the
// linear ops (linear_row_parallel, linear_add), whose public signatures carry only the
// PeerEvents pair; plumbing a second object through every layer would touch a dozen signatures
// to serve one machine-specific transport. The install is exclusive: a second PeerMailbox
// constructor on a different device pair uninstalls the first and takes over, which keeps a
// two-Program process sound (the later Program wins; the earlier one falls back to the staged
// path). Disable entirely with NINFER_TP2_MAILBOX=0 for A/B measurement.
//
// SELECTIVITY (see wants() in allreduce.cu). A captured collective uses the mailbox only when
//   * the mailbox is installed for THIS ExecutionContext,
//   * the caller's stream is capturing (this is a captured decode graph, not an eager pass or
//     a prefill chunk -- prefill's [hidden, chunk] reductions are megabytes, where the staged
//     path's bandwidth beats kernel-driven zero-copy latency, and eager calls have no
//     between-round reset point), and
//   * the payload fits a slot.
// Everything else takes the staged path unchanged, which also means a mailbox-capable Program
// degrades to today's exact behavior the moment any of these predicates fails.

#include "core/device.h" // ExecutionContext
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

class PeerMailbox {
public:
    // Allocates the pinned host slab and the per-slot arrival counters, then installs this
    // instance as the process-wide transport for `ec`'s device pair. Throws on allocation
    // failure, in which case nothing is installed and collectives keep the staged path.
    //
    // `slot_bytes` is the capacity of ONE exchange slot: the largest captured reduction the
    // decode program issues (the [hidden, draft_window+1] MTP-verify activation) plus
    // alignment. Larger captured payloads stay on the staged path.
    // `slots` must cover every captured collective call site (64 layers x 2 projections x the
    // graph families and batch profiles the Program captures), with room to spare.
    PeerMailbox(const ExecutionContext& ec, std::size_t slot_bytes, int slots);
    ~PeerMailbox();

    PeerMailbox(const PeerMailbox&)            = delete;
    PeerMailbox& operator=(const PeerMailbox&) = delete;

    // The mailbox is consulted by the collectives on every captured call; this is the live
    // instance for `ec`, or null when none is. The collectives claim slots through it at
    // capture time (a mutating claim -- the pointer is deliberately non-const), and read the
    // slab's addressing through it afterwards.
    [[nodiscard]] static PeerMailbox* installed(const ExecutionContext& ec) noexcept;

    // True when this mailbox serves `ec`'s exact device pair.
    [[nodiscard]] bool serves(const ExecutionContext& ec) const noexcept;

    // One slot's addressing: the publish area for `rank` and its release flag. Both index by
    // slot: `payload(rank, slot)` is the byte address of that slot's publish area, and
    // `flag(rank, slot)` is the slot's release word.
    [[nodiscard]] void* payload(int rank, int slot) const noexcept;
    [[nodiscard]] volatile std::uint32_t* flag(int rank, int slot) const noexcept;

    // A slot's capacity in bytes (the constructor's slot_bytes, 256-byte aligned).
    [[nodiscard]] std::size_t slot_bytes() const noexcept;

    // The aggregate hang-guard word (host side): the exchange kernels OR a fault into it, the
    // engine reads it once per round. Mutable through the kernel's eyes by design -- it is the
    // pollers' fault report, not engine state.
    [[nodiscard]] volatile std::uint32_t* hang_word() const noexcept;

    // Per-slot block-arrival counters, in the memory of device `rank`.
    [[nodiscard]] std::uint32_t* arrival(int rank) const noexcept;

    // Claims the next free slot, in capture order. Called once per captured collective call
    // site, so a graph replay fires every slot exactly once. A caller that exhausts the slab
    // gets a null and must fall back to the staged path.
    [[nodiscard]] int take_capture_slot() noexcept;

    // Host-side reset of every release flag and hang word, called between graph replays once
    // the round has retired on both devices. Pinned WB words are coherent with the GPUs' PCIe
    // view, so plain stores are the whole protocol.
    static void reset_host_flags() noexcept;

    // True when any exchange's hang guard fired on a previous round: a mailbox poller gave up
    // waiting for its peer. Read once per round, before reset_host_flags(), by the graph launch
    // path; a reported hang is a hard fault (the round's reductions were skipped), not a stall
    // to retry.
    [[nodiscard]] static bool hang_reported() noexcept;

    // Environment override: NINFER_TP2_MAILBOX=0 keeps every collective on the staged path.
    [[nodiscard]] static bool enabled_by_environment() noexcept;

private:
    void* slab_                    = nullptr;  // pinned host allocation, UVA-mapped
    void* payload_[2]             = {nullptr, nullptr};
    std::uint32_t* flags_[2]       = {nullptr, nullptr};
    std::uint32_t* hang_           = nullptr;
    std::uint32_t* arrival_[2]    = {nullptr, nullptr};  // device allocations
    std::size_t slot_bytes_       = 0;
    int slots_                    = 0;
    int taken_                    = 0;
    int devices_[2]               = {0, 0};
    static PeerMailbox* installed_;  // process-wide live instance
};

} // namespace ninfer::ops
