#pragma once

#include "llama-kv-cache.h"
#include "llama-kvarn.h"

#include <memory>
#include <unordered_map>
#include <vector>

struct llama_hparams;
struct llama_model;

class llama_kv_cache_kvarn;

class llama_kv_cache_kvarn_context : public llama_kv_cache_context {
public:
    llama_kv_cache_kvarn_context(
            llama_kv_cache_kvarn * cache,
            llama_memory_context_ptr base);

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    uint32_t get_n_kv() const override;
    llama_kv_cache * get_kv() const override;
    const llama_kv_cache::slot_info & current_sinfo() const override;

    ggml_type type_k() const override;
    ggml_type type_v() const override;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const override;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const override;

    ggml_tensor * get_turbo_rotation() const override;
    ggml_tensor * get_turbo_rotation_inv() const override;
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const override;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const override;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const override;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const override;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const override;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_v_idxs_backend(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_shift(ggml_tensor * dst) const override;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const override;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const override;
    void set_input_k_rot(ggml_tensor * dst) const override;
    void set_input_v_rot(ggml_tensor * dst) const override;
    void set_input_k_rot_backend(ggml_tensor * dst) const override;
    void set_input_v_rot_backend(ggml_tensor * dst) const override;

private:
    llama_kv_cache_context * base() const;

    llama_kv_cache_kvarn * cache;
    llama_memory_context_ptr base_ctx;

    mutable std::unordered_map<int32_t, ggml_tensor *> stored_k;
    mutable std::unordered_map<int32_t, ggml_tensor *> stored_v;
};

class llama_kv_cache_kvarn : public llama_memory_i {
public:
    llama_kv_cache_kvarn(
            const llama_model & model,
            const llama_hparams & hparams,
            llama_kvarn_params params,
            bool offload,
            bool unified,
            uint32_t kv_size,
            uint32_t n_seq_max);

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) override;
    int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_cp_recurrent(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache * get_metadata_cache() const;

    ggml_tensor * store(
            ggml_context * ctx,
            ggml_tensor * current,
            ggml_tensor * indices,
            int32_t il,
            bool value) const;
    ggml_tensor * materialize(
            ggml_context * ctx,
            ggml_tensor * stored,
            int32_t il,
            uint32_t n_kv,
            bool value) const;

private:
    struct layer {
        uint32_t il;
        ggml_tensor * k_records;
        ggml_tensor * v_records;
        ggml_tensor * k_stage;
        ggml_tensor * v_stage;
    };

    const layer & layer_for(int32_t il) const;
    bool can_remove(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;

    const llama_model & model;
    const llama_hparams & hparams;
    const llama_kvarn_params params;

    std::unique_ptr<llama_kv_cache> metadata;
    std::vector<layer> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
};
