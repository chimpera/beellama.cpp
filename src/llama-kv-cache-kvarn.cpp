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
constexpr uint32_t KVAR_N_STATE_VERSION = 2;

bool kvarn_backend_supports_native_ops(ggml_backend_dev_t dev) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return true;
    }

    auto * reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name != nullptr && std::strstr(name, "CUDA") != nullptr;
}

size_t kvarn_record_bytes(int bits) {
    return llama_kvarn_packed_bytes(KVAR_N_GROUP * KVAR_N_GROUP, bits) +
        3 * KVAR_N_GROUP * sizeof(ggml_fp16_t);
}

void write_kvarn_tensor(llama_io_write_i & io, ggml_tensor * tensor) {
    const uint64_t size = ggml_nbytes(tensor);
    io.write(&size, sizeof(size));
    io.write_tensor(tensor, 0, size);
}

void read_kvarn_tensor(llama_io_read_i & io, ggml_tensor * tensor) {
    uint64_t size;
    io.read(&size, sizeof(size));
    if (size != ggml_nbytes(tensor)) {
        throw std::runtime_error("mismatched KVarN cache tensor size");
    }
    io.read_tensor(tensor, 0, size);
}

} // namespace

llama_kv_cache_kvarn_context::llama_kv_cache_kvarn_context(
        llama_kv_cache_kvarn * cache,
        llama_memory_context_ptr base,
        llama_context * update_lctx) :
    llama_kv_cache_context(base ? base->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE),
    cache(cache),
    base_ctx(std::move(base)),
    update_lctx(update_lctx) {
}

llama_kv_cache_context * llama_kv_cache_kvarn_context::base() const {
    return static_cast<llama_kv_cache_context *>(base_ctx.get());
}

bool llama_kv_cache_kvarn_context::next() {
    return base()->next();
}

bool llama_kv_cache_kvarn_context::apply() {
    if (!base()->apply()) {
        return false;
    }

    return !update_lctx || cache->apply_pending_stream_copies(update_lctx);
}

llama_memory_status llama_kv_cache_kvarn_context::get_status() const {
    const auto status = base_ctx ? base_ctx->get_status() : LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    if (status == LLAMA_MEMORY_STATUS_NO_UPDATE && cache->has_pending_stream_copies()) {
        return LLAMA_MEMORY_STATUS_SUCCESS;
    }
    return status;
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
    const auto it = stored_k.find(cache->mapped_layer_id(il));
    GGML_ASSERT(it != stored_k.end());
    return cache->materialize(ctx, it->second, il, get_n_kv(), current_sinfo(), false);
}

ggml_tensor * llama_kv_cache_kvarn_context::get_v(ggml_context * ctx, int32_t il) const {
    const auto it = stored_v.find(cache->mapped_layer_id(il));
    GGML_ASSERT(it != stored_v.end());
    return cache->materialize(ctx, it->second, il, get_n_kv(), current_sinfo(), true);
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
    stored_k[cache->mapped_layer_id(il)] = result;
    return result;
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_v(
        ggml_context * ctx,
        ggml_tensor * v_cur,
        ggml_tensor * v_idxs,
        int32_t il) const {
    auto * result = cache->store(ctx, v_cur, v_idxs, il, true);
    stored_v[cache->mapped_layer_id(il)] = result;
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
        uint32_t n_seq_max,
        uint32_t n_pad,
        uint32_t n_swa,
        llama_swa_type swa_type,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse) :
    model(model),
    hparams(hparams),
    params(params),
    n_stream(unified ? 1u : n_seq_max),
    n_groups_per_stream((kv_size + KVAR_N_GROUP - 1) / KVAR_N_GROUP),
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
        n_pad,
        n_swa,
        swa_type,
        [](int32_t) { return false; },
        nullptr)) {
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(kv_size % KVAR_N_GROUP == 0);

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
            /*.mem_size   =*/ size_t((4u + 4u * n_stream) * hparams.n_layer_kv() * ggml_tensor_overhead()),
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

    const size_t k_record_size = kvarn_record_bytes(params.key_bits);
    const size_t v_record_size = kvarn_record_bytes(params.value_bits);
    const int64_t n_record_groups = int64_t(n_groups_per_stream) * n_stream;
    const int64_t n_stage_tokens = int64_t(KVAR_N_GROUP) * KVAR_N_STAGE_GROUPS * n_stream;
    size_t raw_bytes = 0;

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (!hparams.has_kv(il)) {
            continue;
        }
        if (filter && !filter(il)) {
            continue;
        }

        auto * dev = offload ? model.dev_layer(il) : nullptr;
        if (offload && !kvarn_backend_supports_native_ops(dev)) {
            throw std::runtime_error(format(
                "KVarN cache layer %u is assigned to backend %s, which has no native KVarN operations; "
                "use CUDA or disable KV offload for the CPU fallback",
                il, dev ? ggml_backend_dev_name(dev) : "unknown"));
        }

        auto * buft = offload ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        auto * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create KVarN cache tensor context");
        }

        const uint32_t n_head_kv = hparams.n_head_kv(il);
        const uint32_t head_dim_k = hparams.n_embd_head_k(il);
        const uint32_t head_dim_v = hparams.n_embd_head_v(il);
        const int k_slices = llama_kvarn_head_slices(head_dim_k);
        const int v_slices = llama_kvarn_head_slices(head_dim_v);
        if (k_slices <= 0 || v_slices <= 0) {
            throw std::runtime_error(format(
                "KVarN cache layer %u has unsupported K/V head dimensions %u/%u",
                il, head_dim_k, head_dim_v));
        }

        const uint32_t n_head_k_sliced = n_head_kv * (uint32_t) k_slices;
        const uint32_t n_head_v_sliced = n_head_kv * (uint32_t) v_slices;
        auto * k_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, k_record_size, n_head_k_sliced, n_record_groups);
        auto * v_records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, v_record_size, n_head_v_sliced, n_record_groups);
        auto * k_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_k_sliced, n_stage_tokens);
        auto * v_stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_GROUP, n_head_v_sliced, n_stage_tokens);

        ggml_format_name(k_records, "cache_kvarn_k_records_l%d", il);
        ggml_format_name(v_records, "cache_kvarn_v_records_l%d", il);
        ggml_format_name(k_stage, "cache_kvarn_k_stage_l%d", il);
        ggml_format_name(v_stage, "cache_kvarn_v_stage_l%d", il);

        std::vector<ggml_tensor *> k_records_stream;
        std::vector<ggml_tensor *> v_records_stream;
        std::vector<ggml_tensor *> k_stage_stream;
        std::vector<ggml_tensor *> v_stage_stream;
        k_records_stream.reserve(n_stream);
        v_records_stream.reserve(n_stream);
        k_stage_stream.reserve(n_stream);
        v_stage_stream.reserve(n_stream);

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto * k_records_view = ggml_view_3d(
                    ctx, k_records,
                    k_record_size, n_head_k_sliced, n_groups_per_stream,
                    k_records->nb[1], k_records->nb[2],
                    size_t(s) * n_groups_per_stream * k_records->nb[2]);
            auto * v_records_view = ggml_view_3d(
                    ctx, v_records,
                    v_record_size, n_head_v_sliced, n_groups_per_stream,
                    v_records->nb[1], v_records->nb[2],
                    size_t(s) * n_groups_per_stream * v_records->nb[2]);
            auto * k_stage_view = ggml_view_3d(
                    ctx, k_stage,
                    KVAR_N_GROUP, n_head_k_sliced, KVAR_N_GROUP * KVAR_N_STAGE_GROUPS,
                    k_stage->nb[1], k_stage->nb[2],
                    size_t(s) * KVAR_N_GROUP * KVAR_N_STAGE_GROUPS * k_stage->nb[2]);
            auto * v_stage_view = ggml_view_3d(
                    ctx, v_stage,
                    KVAR_N_GROUP, n_head_v_sliced, KVAR_N_GROUP * KVAR_N_STAGE_GROUPS,
                    v_stage->nb[1], v_stage->nb[2],
                    size_t(s) * KVAR_N_GROUP * KVAR_N_STAGE_GROUPS * v_stage->nb[2]);

            ggml_format_name(k_records_view, "cache_kvarn_k_records_l%d_s%d", il, s);
            ggml_format_name(v_records_view, "cache_kvarn_v_records_l%d_s%d", il, s);
            ggml_format_name(k_stage_view, "cache_kvarn_k_stage_l%d_s%d", il, s);
            ggml_format_name(v_stage_view, "cache_kvarn_v_stage_l%d_s%d", il, s);

            k_records_stream.push_back(k_records_view);
            v_records_stream.push_back(v_records_view);
            k_stage_stream.push_back(k_stage_view);
            v_stage_stream.push_back(v_stage_view);
        }

        map_layer_ids[il] = layers.size();
        layers.push_back({
            il,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            (uint32_t) k_slices,
            (uint32_t) v_slices,
            k_records,
            v_records,
            k_stage,
            v_stage,
            std::move(k_records_stream),
            std::move(v_records_stream),
            std::move(k_stage_stream),
            std::move(v_stage_stream),
        });

        raw_bytes += size_t(kv_size) * n_stream * n_head_kv * (head_dim_k + head_dim_v) * sizeof(ggml_fp16_t);
    }

    if (reuse) {
        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
            const int32_t il_reuse = reuse(il);
            if (il_reuse < 0) {
                continue;
            }
            if (filter && !filter(il)) {
                continue;
            }
            const auto src = map_layer_ids.find(il_reuse);
            if (src == map_layer_ids.end()) {
                throw std::runtime_error(format("KVarN cache layer %u cannot reuse missing layer %d", il, il_reuse));
            }

            const auto & reused = layers.at(src->second);
            if (hparams.n_head_kv(il) != reused.n_head_kv ||
                hparams.n_embd_head_k(il) != reused.head_dim_k ||
                hparams.n_embd_head_v(il) != reused.head_dim_v) {
                throw std::runtime_error(format(
                    "KVarN cache layer %u cannot reuse layer %d with different KV shape",
                    il, il_reuse));
            }

            map_layer_ids[il] = src->second;
        }
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

    LLAMA_LOG_INFO("%s: type = %s, layers = %zu, groups/stream = %u, streams = %u, KVarN = %.2f MiB, equivalent F16 = %.2f MiB\n",
            __func__, llama_kvarn_type_name(params.type), layers.size(), n_groups_per_stream, n_stream,
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
    return std::make_unique<llama_kv_cache_kvarn_context>(this, metadata->init_update(lctx, optimize), lctx);
}

uint32_t llama_kv_cache_kvarn::get_kv_n_stream() const {
    return metadata->get_n_stream();
}

uint32_t llama_kv_cache_kvarn::get_kv_size() const {
    return metadata->get_size();
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_kv_batch(const std::vector<llama_ubatch> & ubatches) {
    auto sinfos = metadata->prepare(ubatches);
    if (sinfos.empty()) {
        return std::make_unique<llama_kv_cache_kvarn_context>(
                this, std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE));
    }

    return std::make_unique<llama_kv_cache_kvarn_context>(
            this, std::make_unique<llama_kv_cache_context>(metadata.get(), std::move(sinfos), ubatches));
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
    const uint32_t stream_src = metadata->get_stream_for_seq(seq_id_src);
    const uint32_t stream_dst = metadata->get_stream_for_seq(seq_id_dst);

    if (stream_src != stream_dst) {
        bool is_full = true;

        if (p0 > 0 && p0 + 1 < (int) get_kv_size()) {
            is_full = false;
        }

        if (p1 > 0 && p1 + 1 < (int) get_kv_size()) {
            is_full = false;
        }

        GGML_ASSERT(is_full && "KVarN cross-stream seq_cp() is only supported for full KV buffers");

        pending_stream_copies.ssrc.push_back(stream_src);
        pending_stream_copies.sdst.push_back(stream_dst);
    }

    metadata->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_kvarn::seq_keep(llama_seq_id seq_id) {
    metadata->seq_keep(seq_id);
}

void llama_kv_cache_kvarn::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) {
    GGML_ABORT("KVarN does not support position shifts");
}

void llama_kv_cache_kvarn::seq_div(llama_seq_id, llama_pos, llama_pos, int) {
    GGML_ABORT("KVarN does not support position division");
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

bool llama_kv_cache_kvarn::has_pending_stream_copies() const {
    return !pending_stream_copies.empty();
}

void llama_kv_cache_kvarn::copy_kvarn_stream(uint32_t stream_src, uint32_t stream_dst) {
    GGML_ASSERT(stream_src < n_stream);
    GGML_ASSERT(stream_dst < n_stream);
    GGML_ASSERT(stream_src != stream_dst);

    LLAMA_LOG_DEBUG("%s: copying KVarN stream %u to stream %u\n", __func__, stream_src, stream_dst);

    for (auto & layer : layers) {
        ggml_backend_tensor_copy(layer.k_records_stream[stream_src], layer.k_records_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.v_records_stream[stream_src], layer.v_records_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.k_stage_stream[stream_src], layer.k_stage_stream[stream_dst]);
        ggml_backend_tensor_copy(layer.v_stage_stream[stream_src], layer.v_stage_stream[stream_dst]);
    }
}

bool llama_kv_cache_kvarn::apply_pending_stream_copies(llama_context * lctx) {
    if (pending_stream_copies.empty()) {
        return true;
    }

    GGML_ASSERT(pending_stream_copies.ssrc.size() == pending_stream_copies.sdst.size());
    llama_synchronize(lctx);

    const size_t n_copy = pending_stream_copies.ssrc.size();
    for (size_t i = 0; i < n_copy; ++i) {
        copy_kvarn_stream(pending_stream_copies.ssrc[i], pending_stream_copies.sdst[i]);
    }

    pending_stream_copies.ssrc.clear();
    pending_stream_copies.sdst.clear();
    return true;
}

void llama_kv_cache_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    metadata->state_write(io, seq_id, flags);

    std::vector<uint32_t> saved_streams;
    if (seq_id == -1) {
        saved_streams.reserve(n_stream);
        for (uint32_t stream = 0; stream < n_stream; ++stream) {
            saved_streams.push_back(stream);
        }
    } else {
        const uint32_t stream = metadata->get_stream_for_seq(seq_id);
        GGML_ASSERT(stream < n_stream);
        saved_streams.push_back(stream);
    }

    io.write(&KVAR_N_STATE_MAGIC, sizeof(KVAR_N_STATE_MAGIC));
    io.write(&KVAR_N_STATE_VERSION, sizeof(KVAR_N_STATE_VERSION));
    const int32_t type = params.type;
    const uint32_t n_layers = layers.size();
    const uint32_t n_saved_streams = saved_streams.size();
    io.write(&type, sizeof(type));
    io.write(&n_layers, sizeof(n_layers));
    io.write(&n_saved_streams, sizeof(n_saved_streams));
    for (const uint32_t stream : saved_streams) {
        io.write(&stream, sizeof(stream));
    }

    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));
        for (const uint32_t stream : saved_streams) {
            io.write(&stream, sizeof(stream));
            for (auto * tensor : {
                    layer.k_records_stream[stream],
                    layer.v_records_stream[stream],
                    layer.k_stage_stream[stream],
                    layer.v_stage_stream[stream],
                }) {
                write_kvarn_tensor(io, tensor);
            }
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
    if (magic != KVAR_N_STATE_MAGIC || (version != 1 && version != KVAR_N_STATE_VERSION) ||
        type != params.type || n_layers != layers.size()) {
        throw std::runtime_error("incompatible KVarN cache state");
    }

    if (version == 1) {
        for (const auto & layer : layers) {
            uint32_t il;
            io.read(&il, sizeof(il));
            if (il != layer.il) {
                throw std::runtime_error("mismatched KVarN cache layer");
            }

            for (auto * tensor : { layer.k_records, layer.v_records, layer.k_stage, layer.v_stage }) {
                read_kvarn_tensor(io, tensor);
            }
        }
        return;
    }

    uint32_t n_saved_streams;
    io.read(&n_saved_streams, sizeof(n_saved_streams));
    if (n_saved_streams == 0 || n_saved_streams > n_stream) {
        throw std::runtime_error("invalid KVarN cache stream count");
    }

    std::vector<uint32_t> saved_streams(n_saved_streams);
    for (uint32_t & stream : saved_streams) {
        io.read(&stream, sizeof(stream));
        if (stream >= n_stream) {
            throw std::runtime_error("invalid KVarN cache stream");
        }
    }

    const uint32_t seq_stream = seq_id == -1 ? 0 : metadata->get_stream_for_seq(seq_id);
    if (seq_id != -1 && seq_stream >= n_stream) {
        throw std::runtime_error("invalid KVarN sequence stream");
    }

    for (const auto & layer : layers) {
        uint32_t il;
        io.read(&il, sizeof(il));
        if (il != layer.il) {
            throw std::runtime_error("mismatched KVarN cache layer");
        }

        for (uint32_t i = 0; i < n_saved_streams; ++i) {
            uint32_t stream;
            io.read(&stream, sizeof(stream));
            if (stream != saved_streams[i]) {
                throw std::runtime_error("mismatched KVarN cache stream");
            }

            const uint32_t stream_dst = seq_id == -1 ? stream : seq_stream;
            for (auto * tensor : {
                    layer.k_records_stream[stream_dst],
                    layer.v_records_stream[stream_dst],
                    layer.k_stage_stream[stream_dst],
                    layer.v_stage_stream[stream_dst],
                }) {
                read_kvarn_tensor(io, tensor);
            }
        }
    }
}

llama_kv_cache * llama_kv_cache_kvarn::get_metadata_cache() const {
    return metadata.get();
}

int32_t llama_kv_cache_kvarn::mapped_layer_id(int32_t il) const {
    return map_layer_ids.at(il);
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

    const uint32_t head_dim = value ? layer.head_dim_v : layer.head_dim_k;
    const uint32_t slices = value ? layer.v_slices : layer.k_slices;
    GGML_ASSERT((uint32_t) current->ne[0] == head_dim);
    GGML_ASSERT((uint32_t) current->ne[1] == layer.n_head_kv);
    if (slices > 1) {
        current = ggml_reshape_3d(ctx, current, KVAR_N_GROUP, layer.n_head_kv * slices, current->ne[2]);
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
        const llama_kv_cache::slot_info & sinfo,
        bool value) const {
    const auto & layer = layer_for(il);
    const uint32_t stream_start = sinfo.s0;
    const uint32_t stream_count = sinfo.s1 - sinfo.s0 + 1;

    ggml_tensor * result = ggml_kvarn_materialize(
        ctx,
        value ? layer.v_records : layer.k_records,
        stored,
        stored->src[1],
        n_kv,
        stream_start,
        stream_count,
        value ? params.value_bits : params.key_bits,
        value);

    const uint32_t slices = value ? layer.v_slices : layer.k_slices;
    if (slices > 1) {
        result = ggml_reshape_4d(
                ctx,
                result,
                value ? layer.head_dim_v : layer.head_dim_k,
                layer.n_head_kv,
                n_kv,
                stream_count);
    }

    return result;
}
