// ===========================================================================
// PROVENANCE: ported from github.com/ivanov84/ninfer-windows-tp2
//             (branch windows-tp2). Upstream lineage: NInfer TP2.
//             See LICENSE_ref for the license terms.
//             Ported on 2026-09-20 to give this tree the pinned-host peer
//             mailbox transport for Windows TP2 collectives.
//             Content kept verbatim apart from this banner.
// ===========================================================================
// Implements: include/ninfer/ops/peer_mailbox.h
//
// The pinned host slab layout, in one allocation:
//
//   [ payload rank0 slot0 .. slotN-1 ][ payload rank1 slot0 .. slotN-1 ]
//   [ flags rank0 slot0..N-1 ][ flags rank1 slot0..N-1 ][ hang word ]
//
// Payload slots are 256-byte aligned (the vectorized exchange reads and writes 16-byte units;
// the headroom also keeps a slot's stores on distinct cache lines). Flag words sit after the
// payload so a payload overflow from a mis-sized slot cannot reach them without being loudly
// out of contract, and the hang word is last.
//
// The per-slot arrival counters are per-DEVICE allocations (VRAM): each rank's kernel uses only
// its own device's counter, which keeps them inside one device's memory model.

#include "ninfer/ops/peer_mailbox.h"

#include "core/device.h" // CUDA_CHECK

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

PeerMailbox* PeerMailbox::installed_ = nullptr;

namespace {

constexpr std::size_t kSlotAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kSlotAlignment - 1) & ~(kSlotAlignment - 1);
}

// Current-device save/restore for the two phases that need a device current: allocating each
// rank's arrival counters in that rank's context, and freeing them again.
class ScopedDevice {
public:
    ScopedDevice() { CUDA_CHECK(cudaGetDevice(&previous_)); }
    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

    static void set(int device) { CUDA_CHECK(cudaSetDevice(device)); }

private:
    int previous_ = 0;
};

void require_two_devices(const ExecutionContext& ec, const char* message) {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value() ||
        ec.dev[0]->device == ec.dev[1]->device) {
        throw std::invalid_argument(message);
    }
}

} // namespace

PeerMailbox::PeerMailbox(const ExecutionContext& ec, std::size_t slot_bytes, int slots)
    : slot_bytes_(aligned(slot_bytes)), slots_(slots) {
    require_two_devices(ec, "PeerMailbox: requires an ExecutionContext with two distinct devices");
    if (slots < 1) { throw std::invalid_argument("PeerMailbox: requires at least one slot"); }
    if ((slot_bytes_ % 16) != 0) {
        throw std::invalid_argument("PeerMailbox: slot bytes must cover whole 16-byte vectors");
    }

    const int pair[2] = {ec.dev[0]->device, ec.dev[1]->device};
    devices_[0]       = pair[0];
    devices_[1]       = pair[1];

    const std::size_t payload_bytes = static_cast<std::size_t>(slots_) * slot_bytes_ * 2;
    const std::size_t words_bytes =
        (static_cast<std::size_t>(slots_) * 2 + 1) * sizeof(std::uint32_t);
    const std::size_t total = payload_bytes + words_bytes + kSlotAlignment;

    // Everything allocates into locals first and commits to members only when the whole set
    // succeeded: a throwing constructor does not run the destructor, so a half-built object
    // must leave nothing behind that needs it.
    void* slab                 = nullptr;
    std::uint32_t* arrivals[2] = {nullptr, nullptr};
    const cudaError_t status   = cudaHostAlloc(&slab, total, cudaHostAllocMapped);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("PeerMailbox: cudaHostAlloc failed: ") +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    }
    try {
        std::memset(slab, 0, total);
        const ScopedDevice scope;
        const std::size_t counters = static_cast<std::size_t>(slots_) * sizeof(std::uint32_t);
        for (int rank = 0; rank < 2; ++rank) {
            ScopedDevice::set(pair[rank]);
            const cudaError_t alloc = cudaMalloc(&arrivals[rank], counters);
            if (alloc != cudaSuccess) {
                throw std::runtime_error(std::string("PeerMailbox: arrival counter allocation "
                                                     "failed: ") +
                                         cudaGetErrorName(alloc) + ": " +
                                         cudaGetErrorString(alloc));
            }
            const cudaError_t zero = cudaMemset(arrivals[rank], 0, counters);
            if (zero != cudaSuccess) {
                throw std::runtime_error(std::string("PeerMailbox: arrival counter zeroing "
                                                     "failed: ") +
                                         cudaGetErrorName(zero) + ": " + cudaGetErrorString(zero));
            }
        }
    } catch (...) {
        (void)cudaFreeHost(slab);
        const ScopedDevice scope;
        for (int rank = 0; rank < 2; ++rank) {
            if (arrivals[rank] != nullptr) {
                (void)cudaSetDevice(pair[rank]);
                (void)cudaFree(arrivals[rank]);
            }
        }
        throw;
    }

    slab_       = slab;
    auto* base  = static_cast<std::uint8_t*>(slab_);
    payload_[0] = base;
    payload_[1] = base + static_cast<std::size_t>(slots_) * slot_bytes_;
    auto* words = reinterpret_cast<std::uint32_t*>(base + payload_bytes);
    flags_[0]   = words;
    flags_[1]   = flags_[0] + slots_;
    hang_       = flags_[1] + slots_;
    arrival_[0] = arrivals[0];
    arrival_[1] = arrivals[1];

    installed_ = this;
}

PeerMailbox::~PeerMailbox() {
    if (installed_ == this) { installed_ = nullptr; }
    if (slab_ != nullptr) {
        const ScopedDevice scope;
        (void)cudaFreeHost(slab_);
        for (int rank = 0; rank < 2; ++rank) {
            if (arrival_[rank] != nullptr) {
                (void)cudaSetDevice(devices_[rank]);
                (void)cudaFree(arrival_[rank]);
            }
        }
    }
    slab_       = nullptr;
    payload_[0] = nullptr;
    payload_[1] = nullptr;
    flags_[0]   = nullptr;
    flags_[1]   = nullptr;
    hang_       = nullptr;
    arrival_[0] = nullptr;
    arrival_[1] = nullptr;
    taken_      = 0;
}

PeerMailbox* PeerMailbox::installed(const ExecutionContext& ec) noexcept {
    if (installed_ == nullptr) { return nullptr; }
    return installed_->serves(ec) ? installed_ : nullptr;
}

bool PeerMailbox::serves(const ExecutionContext& ec) const noexcept {
    if (ec.tp != 2 || !ec.dev[0].has_value() || !ec.dev[1].has_value()) { return false; }
    return ec.dev[0]->device == devices_[0] && ec.dev[1]->device == devices_[1];
}

void* PeerMailbox::payload(int rank, int slot) const noexcept {
    return static_cast<std::uint8_t*>(payload_[rank]) +
           static_cast<std::size_t>(slot) * slot_bytes_;
}

volatile std::uint32_t* PeerMailbox::flag(int rank, int slot) const noexcept {
    return flags_[rank] + slot;
}

std::size_t PeerMailbox::slot_bytes() const noexcept { return slot_bytes_; }

volatile std::uint32_t* PeerMailbox::hang_word() const noexcept { return hang_; }

std::uint32_t* PeerMailbox::arrival(int rank) const noexcept { return arrival_[rank]; }

int PeerMailbox::take_capture_slot() noexcept {
    if (taken_ >= slots_) { return -1; }
    return taken_++;
}

void PeerMailbox::reset_host_flags() noexcept {
    if (installed_ == nullptr) { return; }
    for (int slot = 0; slot < installed_->slots_; ++slot) {
        installed_->flags_[0][slot] = 0;
        installed_->flags_[1][slot] = 0;
    }
    *installed_->hang_ = 0;
}

bool PeerMailbox::hang_reported() noexcept {
    return installed_ != nullptr && *installed_->hang_ != 0;
}

bool PeerMailbox::enabled_by_environment() noexcept {
    const char* override_value = std::getenv("NINFER_TP2_MAILBOX");
    if (override_value == nullptr) { return true; }
    return std::strcmp(override_value, "0") != 0 && std::strcmp(override_value, "false") != 0;
}

} // namespace ninfer::ops
