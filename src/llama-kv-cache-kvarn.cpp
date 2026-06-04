#include "llama-kv-cache-kvarn.h"

#include "llama-context.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace {

constexpr uint32_t KVAR_N_GROUP = 128;
constexpr uint32_t KVAR_N_STAGE_GROUPS = 3;
constexpr uint32_t KVAR_N_STATE_MAGIC = 0x4e52564b; // "KVRN"
constexpr uint32_t KVAR_N_STATE_VERSION = 1;

bool kvarn_backend_supports_native_ops(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }

    auto * reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name != nullptr &&
        (std::strstr(name, "CUDA") != nullptr ||
         std::strstr(name, "ROCm") != nullptr ||
         std::strstr(name, "HIP") != nullptr ||
         std::strstr(name, "MUSA") != nullptr);
}

size_t kvarn_record_bytes(int bits) {
    return llama_kvarn_packed_bytes(KVAR_N_GROUP * KVAR_N_GROUP, bits) +
        3 * KVAR_N_GROUP * sizeof(ggml_fp16_t);
}

} // namespace

llama_kv_cache_kvarn_context::llama_kv_cache_kvarn_context(
        llama_kv_cache_kvarn * cache,
        llama_memory_context_ptr base) :
    llama_kv_cache_context(base ? base->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE),
    cache(cache),
    base_ctx(std::move(base)) {
}

llama_kv_cache_context * llama_kv_cache_kvarn_context::base() const {
    return static_cast<llama_kv_cache_context *>(base_ctx.get());
}

bool llama_kv_cache_kvarn_context::next() {
    return base()->next();
}

bool llama_kv_cache_kvarn_context::apply() {
    return base()->apply();
}

llama_memory_status llama_kv_cache_kvarn_context::get_status() const {
    return base_ctx ? base_ctx->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE;
}

const llama_ubatch & llama_kv_cache_kvarn_context::get_ubatch() const {
    return base()->get_ubatch();
}

uint32_t llama_kv_cache_kvarn_context::get_n_kv() const {
    return base()->get_n_kv();
}

llama_kv_cache * llama_kv_cache_kvarn_context::get_kv() const {
    return cache->get_metadata_cache();
}

const llama_kv_cache::slot_info & llama_kv_cache_kvarn_context::current_sinfo() const {
    return base()->current_sinfo();
}

ggml_type llama_kv_cache_kvarn_context::type_k() const {
    return GGML_TYPE_F16;
}

ggml_type llama_kv_cache_kvarn_context::type_v() const {
    return GGML_TYPE_F16;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_k(ggml_context * ctx, int32_t il) const {
    const auto it = stored_k.find(il);
    GGML_ASSERT(it != stored_k.end());
    return cache->materialize(ctx, it->second, il, get_n_kv(), false);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v(ggml_context * ctx, int32_t il) const {
    const auto it = stored_v.find(il);
    GGML_ASSERT(it != stored_v.end());
    return cache->materialize(ctx, it->second, il, get_n_kv(), true);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rotation() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rotation_inv() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rot_forward() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::get_turbo_rot_inverse() const {
    return nullptr;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_k(
        ggml_context * ctx,
        ggml_tensor * k_cur,
        ggml_tensor * k_idxs,
        int32_t il) const {
    auto * result = cache->store(ctx, k_cur, k_idxs, il, false);
    stored_k[il] = result;
    return result;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v(
        ggml_context * ctx,
        ggml_tensor * v_cur,
        ggml_tensor * v_idxs,
        int32_t il) const {
    auto * result = cache->store(ctx, v_cur, v_idxs, il, true);
    stored_v[il] = result;
    return result;
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return base()->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_k_rot(ggml_context * ctx) const {
    return base()->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_v_rot(ggml_context * ctx) const {
    return base()->build_input_v_rot(ctx);
}

void llama_kv_cache_kvarn_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_k_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_v_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_k_idxs_backend(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_v_idxs_backend(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_shift(ggml_tensor * dst) const {
    base()->set_input_k_shift(dst);
}

void llama_kv_cache_kvarn_context::set_input_kq_mask(
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        bool causal_attn) const {
    base()->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_kvarn_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    base()->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_k_rot(ggml_tensor * dst) const {
    base()->set_input_k_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot(ggml_tensor * dst) const {
    base()->set_input_v_rot(dst);
}

void llama_kv_cache_kvarn_context::set_input_k_rot_backend(ggml_tensor * dst) const {
    base()->set_input_k_rot_backend(dst);
}

void llama_kv_cache_kvarn_context::set_input_v_rot_backend(ggml_tensor * dst) const {
    base()->set_input_v_rot_backend(dst);
}

llama_kv_cache_kvarn::llama_kv_cache_kvarn(
        const llama_model & model,
        const llama_hparams & hparams,
        llama_kvarn_params params,
        bool offload,
        bool unified,
        uint32_t kv_size,
        uint32_t n_seq_max) :
    model(model),
    hparams(hparams),
    params(params),
    metadata(std::make_unique<llama_kv_cache>(
        model,
        hparams,
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        false,
        false,
        unified,
        kv_size,
        n_seq_max,
        1,
        0,
        LLAMA_SWA_TYPE_NONE,
        [](int32_t) { return false; },
        nullptr)) {
    struct buft_comparator {
        bool operator()(ggml_backend_buffer_type_t lhs, ggml_backend_buffer_type_t rhs) const {
            return std::strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        const auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) {
            return it->second.get();
        }

        ggml_init_params ctx_params = {
            /*.mem_size   =*/ size_t(4 * hparams.n_layer_kv() * ggml_tensor_overhead()),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx { ggml_init(ctx_params) };
        if (!ctx) {
            return nullptr;
        }

        auto * result = ctx.get();
        ctx_map.emplace(buft, std::move(ctx));
        return result;
    };

    const uint32_t n_groups = (kv_size + KVAR_N_GROUP - 1) / KVAR_N_GROUP;
    const size_t k_record_size = kvarn_record_bytes(params.key_bits);
    const size_t v_record_size = kvarn_record_bytes(params.value_bits);

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (!hparams.has_kv(il)) {
            continue;
        }

        auto * dev = offload ? model.dev_layer(il) : nullptr;
        if (offload && !kvarn_backend_supports_native_ops(dev)) {
            throw std::runtime_error(format(
                "KVarN cache layer %u is assigned to backend %s, which has no native KVarN operations; "
                "use CUDA/ROCm or disable KV offload for the CPU fallback",
                il, dev ? ggml_backend_dev_name(dev) : "unknown"));
        }

        auto * buft = offload ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        auto * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create KVarN cache tensor context");
        }

        const uint32_t n_head_kv = hparams.n_head_kv(il);
        auto * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_size, n_head_kv, n_groups);
        auto * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_size, n_head_kv, n_groups);
        auto * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_kv, KVAR_N_GROUP * KVAR_N_STAGE_GROUPS);
        auto * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_kv, KVAR_N_GROUP * KVAR_N_STAGE_GROUPS);

        ggml_format_name(k_records, "cache_kvarn_k_records_l%d", il);
        ggml_format_name(v_records, "cache_kvarn_v_records_l%d", il);
        ggml_format_name(k_stage, "cache_kvarn_k_stage_l%d", il);
        ggml_format_name(v_stage, "cache_kvarn_v_stage_l%d", il);

        map_layer_ids[il] = layers.size();
        layers.push_back({ il, k_records, v_records, k_stage, v_stage });
    }

    size_t total_bytes = 0;
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (auto * tensor = ggml_get_first_tensor(ctx.get()); tensor != nullptr; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
                tensor->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate KVarN cache buffer");
        }

        ggml_backend_buffer_clear(buf, 0);
        total_bytes += ggml_backend_buffer_get_size(buf);
        LLAMA_LOG_INFO("%s: %10s KVarN buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    const size_t raw_bytes = size_t(kv_size) * hparams.n_layer_kv() *
        hparams.n_head_kv() * KVAR_N_GROUP * 2 * sizeof(ggml_fp16_t);
    LLAMA_LOG_INFO("%s: type = %s, layers = %zu, groups = %u, KVarN = %.2f MiB, equivalent F16 = %.2f MiB\n",
            __func__, llama_kvarn_type_name(params.type), layers.size(), n_groups,
            total_bytes / 1024.0 / 1024.0, raw_bytes / 1024.0 / 1024.0);
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    return std::make_unique<llama_kv_cache_kvarn_context>(
        this, metadata->init_batch(balloc, n_ubatch, embd_all));
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_full() {
    return std::make_unique<llama_kv_cache_kvarn_context>(this, metadata->init_full());
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_update(llama_context * lctx, bool optimize) {
    return metadata->init_update(lctx, optimize);
}

bool llama_kv_cache_kvarn::get_can_shift() const {
    return false;
}

void llama_kv_cache_kvarn::clear(bool data) {
    metadata->clear(false);
    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache_kvarn::can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (p0 < 0 && p1 < 0) {
        return true;
    }
    if (seq_id < 0) {
        return false;
    }

    const llama_pos pos_max = metadata->seq_pos_max(seq_id);
    if (pos_max < 0) {
        return true;
    }

    const llama_pos begin = std::max<llama_pos>(p0, 0);
    const llama_pos end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if (end <= pos_max) {
        return false;
    }

    const llama_pos live_group = pos_max / KVAR_N_GROUP;
    const llama_pos earliest_exact = std::max<llama_pos>(0, live_group - 1) * KVAR_N_GROUP;
    return begin >= earliest_exact;
}

bool llama_kv_cache_kvarn::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!can_remove(seq_id, p0, p1)) {
        LLAMA_LOG_WARN("%s: KVarN can only remove a complete sequence or the current/previous fp16 tail groups\n", __func__);
        return false;
    }
    return metadata->seq_rm(seq_id, p0, p1);
}

bool llama_kv_cache_kvarn::seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) {
    const llama_pos pos_max = metadata->seq_pos_max(seq_id);
    if (pos_max >= 0) {
        const uint32_t earliest_exact = uint32_t(std::max<llama_pos>(0, pos_max / KVAR_N_GROUP - 1) * KVAR_N_GROUP);
        if (cell_idx < earliest_exact) {
            return false;
        }
    }
    return metadata->seq_rm_cell(seq_id, cell_idx);
}

int llama_kv_cache_kvarn::cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) {
    return metadata->cells_at_pos(seq_id, pos, cell_indices, n_max);
}

void llama_kv_cache_kvarn::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    metadata->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_kvarn::seq_keep(llama_seq_id seq_id) {
    metadata->seq_keep(seq_id);
}

void llama_kv_cache_kvarn::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
    LLAMA_LOG_ERROR("%s: KVarN does not support position shifts\n", __func__);
}

void llama_kv_cache_kvarn::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
    LLAMA_LOG_ERROR("%s: KVarN does not support position division\n", __func__);
}

llama_pos llama_kv_cache_kvarn::seq_pos_min(llama_seq_id seq_id) const {
    return metadata->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_kvarn::seq_pos_max(llama_seq_id seq_id) const {
    return metadata->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kvarn::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        auto * buft = ggml_backend_buffer_get_type(buf.get());
        result[buft] += hparams.no_alloc
            ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft)
            : ggml_backend_buffer_get_size(buf.get());
    }
    return result;
}

void llama_kv_cache_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    metadata->state_write(io, seq_id, flags);

    io.write(&KVAR_N_STATE_MAGIC, sizeof(KVAR_N_STATE_MAGIC));
    io.write(&KVAR_N_STATE_VERSION, sizeof(KVAR_N_STATE_VERSION));
    const int32_t type = params.type;
    const uint32_t n_layers = layers.size();
    io.write(&type, sizeof(type));
    io.write(&n_layers, sizeof(n_layers));

    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));
        for (auto * tensor : { layer.k_records, layer.v_records, layer.k_stage, layer.v_stage }) {
            const uint64_t size = ggml_nbytes(tensor);
            io.write(&size, sizeof(size));
            io.write_tensor(tensor, 0, size);
        }
    }
}

void llama_kv_cache_kvarn::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    metadata->state_read(io, seq_id, flags);

    uint32_t magic;
    uint32_t version;
    int32_t type;
    uint32_t n_layers;
    io.read(&magic, sizeof(magic));
    io.read(&version, sizeof(version));
    io.read(&type, sizeof(type));
    io.read(&n_layers, sizeof(n_layers));
    if (magic != KVAR_N_STATE_MAGIC || version != KVAR_N_STATE_VERSION ||
        type != params.type || n_layers != layers.size()) {
        throw std::runtime_error("incompatible KVarN cache state");
    }

    for (const auto & layer : layers) {
        uint32_t il;
        io.read(&il, sizeof(il));
        if (il != layer.il) {
            throw std::runtime_error("mismatched KVarN cache layer");
        }

        for (auto * tensor : { layer.k_records, layer.v_records, layer.k_stage, layer.v_stage }) {
            uint64_t size;
            io.read(&size, sizeof(size));
            if (size != ggml_nbytes(tensor)) {
                throw std::runtime_error("mismatched KVarN cache tensor size");
            }
            io.read_tensor(tensor, 0, size);
        }
    }
}

llama_kv_cache * llama_kv_cache_kvarn::get_metadata_cache() const {
    return metadata.get();
}

const llama_kv_cache_kvarn::layer & llama_kv_cache_kvarn::layer_for(int32_t il) const {
    return layers.at(map_layer_ids.at(il));
}

ggml_tensor * llama_kv_cache_kvarn::store(
        ggml_context * ctx,
        ggml_tensor * current,
        ggml_tensor * indices,
        int32_t il,
        bool value) const {
    const auto & layer = layer_for(il);
    if (!ggml_is_contiguous(current)) {
        current = ggml_cont(ctx, current);
    }

    return ggml_kvarn_store(
        ctx,
        current,
        indices,
        value ? layer.v_stage : layer.k_stage,
        value ? layer.v_records : layer.k_records,
        value ? params.value_bits : params.key_bits,
        params.sinkhorn_iters,
        value);
}

ggml_tensor * llama_kv_cache_kvarn::materialize(
        ggml_context * ctx,
        ggml_tensor * stored,
        int32_t il,
        uint32_t n_kv,
        bool value) const {
    const auto & layer = layer_for(il);
    return ggml_kvarn_materialize(
        ctx,
        value ? layer.v_records : layer.k_records,
        stored,
        stored->src[1],
        n_kv,
        value ? params.value_bits : params.key_bits,
        value);
}
