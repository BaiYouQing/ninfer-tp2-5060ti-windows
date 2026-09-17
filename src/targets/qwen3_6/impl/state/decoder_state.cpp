#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// 单侧校验：bf16 不允许带 quant group；I8 必须每 kKvQuantGroup 一组、fp8 必须每
// kKvFp8ScaleGroup 维一个 fp16 scale。返回该侧是否需要 scale plane。
bool validate_kv_side(DType dtype, std::int32_t quant_group, std::int32_t head_dim,
                      const char* side) {
    switch (dtype) {
    case DType::BF16:
        if (quant_group != 0) {
            throw std::invalid_argument(std::string("Paged KV ") + side +
                                        " bf16 must not carry a quant group");
        }
        return false;
    case DType::I8:
    case DType::FP8_E4M3FN: {
        const std::int32_t expected = dtype == DType::I8 ? kKvQuantGroup : kKvFp8ScaleGroup;
        if (quant_group != expected || head_dim % quant_group != 0) {
            throw std::invalid_argument(std::string("Paged KV ") + side +
                                        " quant group does not match its dtype");
        }
        return true;
    }
    default:
        throw std::invalid_argument(std::string("Paged KV ") + side + " dtype is unsupported");
    }
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType k_dtype,
                              DType v_dtype, std::int32_t k_quant_group, std::int32_t v_quant_group,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool k_scaled = validate_kv_side(k_dtype, k_quant_group, head_dim, "K");
    const bool v_scaled = validate_kv_side(v_dtype, v_quant_group, head_dim, "V");

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;
    // 每层 plane 顺序固定 [K code, V code, K scale?, V scale?]：单侧档位（bf16 / int8）逐 plane
    // 与改造前完全一致；k16v8 = K bf16（无 scale）+ V fp8（每 256 维 1 个 scale）。
    const std::size_t per_layer = 2ULL + (k_scaled ? 1ULL : 0ULL) + (v_scaled ? 1ULL : 0ULL);
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) * per_layer);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        pool_spec.planes.push_back({k_dtype, head_dim, kv_heads, 256});
        pool_spec.planes.push_back({v_dtype, head_dim, kv_heads, 256});
        if (k_scaled) {
            pool_spec.planes.push_back({DType::FP16, head_dim / k_quant_group, kv_heads, 256});
        }
        if (v_scaled) {
            pool_spec.planes.push_back({DType::FP16, head_dim / v_quant_group, kv_heads, 256});
        }
    }
    return PagedKVCacheLayout{
        .pool          = plan_paged_kv_pool(builder, pool_spec),
        .layers        = layers,
        .max_context   = capacity,
        .kv_heads      = kv_heads,
        .head_dim      = head_dim,
        .k_dtype       = k_dtype,
        .v_dtype       = v_dtype,
        .k_quant_group = k_quant_group,
        .v_quant_group = v_quant_group,
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_k_dtype, spec.kv_v_dtype,
                                spec.kv_k_quant_group, spec.kv_v_quant_group, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        // TEMP(diag): MTP 层缓存强制 bf16 —— 用于定位 fp8 档下起草接受率崩塌是否来自 MTP 层。
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, DType::BF16, DType::BF16, 0, 0,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), k_dtype_(layout.k_dtype),
      v_dtype_(layout.v_dtype), k_quant_group_(layout.k_quant_group),
      v_quant_group_(layout.v_quant_group) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool k_scaled      = k_quant_group_ != 0;
    const bool v_scaled      = v_quant_group_ != 0;
    const std::size_t stride = 2ULL + (k_scaled ? 1ULL : 0ULL) + (v_scaled ? 1ULL : 0ULL);
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = k_scaled ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = v_scaled ? pool_.plane(base + 2 + (k_scaled ? 1ULL : 0ULL)) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .k_dtype       = k_dtype_,
        .v_dtype       = v_dtype_,
        .k_quant_group = k_quant_group_,
        .v_quant_group = v_quant_group_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool k_scaled      = k_quant_group_ != 0;
    const bool v_scaled      = v_quant_group_ != 0;
    const std::size_t stride = 2ULL + (k_scaled ? 1ULL : 0ULL) + (v_scaled ? 1ULL : 0ULL);
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = k_scaled ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = v_scaled ? pool_.plane(base + 2 + (k_scaled ? 1ULL : 0ULL)) : Tensor(),
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .k_dtype       = k_dtype_,
        .v_dtype       = v_dtype_,
        .k_quant_group = k_quant_group_,
        .v_quant_group = v_quant_group_,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
