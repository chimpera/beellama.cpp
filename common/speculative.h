#pragma once

#include "llama.h"
#include "common.h"

#include <unordered_map>
#include <vector>

struct common_speculative;

struct common_dflash_ring_write {
    int ring_pos;
    int n_tokens;
    int src_token_offset;
};

struct common_dflash_ring_stats {
    int ring_write_pos = 0;
    int ring_filled    = 0;
    int committed_len  = 0;
    int cross_ctx      = 0;
    int cross_len      = 0;
};

struct common_dflash_prefill_span {
    bool should_flush = false;

    int32_t capture_begin = 0;
    int32_t capture_end   = 0;

    int src_offset = 0;
    int n_tokens   = 0;
};

struct common_speculative_tree {
    std::vector<llama_token> tokens;
    std::vector<int32_t>     parents;
    std::vector<int32_t>     depths;
    std::vector<std::unordered_map<llama_token, int>> child_maps;
    std::vector<uint8_t>     visibility;
    std::vector<float>       log_probs;
    int n_nodes = 0;
    int main_path_len = 0;
};

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;
    int32_t n_min = 0;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    std::vector<float> * draft_log_probs = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target nextn embeddings to be extracted
bool common_speculative_need_embd_nextn(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

common_dflash_ring_write common_dflash_ring_write_plan(int ring_size, int ring_pos, int n_tokens);

int common_dflash_prefill_committed_after_flush(
        int current_committed,
        int n_written,
        int commit_end_pos);

bool common_dflash_prefill_capture_complete_for_test(int captured, int requested);

bool common_dflash_cpu_ring_valid_after_write_for_test(
        bool was_valid,
        bool force_cpu_ring,
        bool has_gpu_ring,
        bool cpu_ring_written_all);

bool common_dflash_should_refuse_large_prefill_fallback_for_test(
        int requested,
        int captured,
        bool use_prefill_gpu,
        bool has_gpu_ring);

bool common_dflash_cpu_ring_valid_after_source_write_for_test(
        bool was_valid,
        int  source,
        bool force_cpu_ring,
        bool has_gpu_ring,
        bool cpu_data_all_layers);

bool common_dflash_tree_update_requires_cpu_hidden_for_test(
        bool has_cpu_hidden,
        bool has_gpu_ring);

int common_dflash_invalid_reduced_logits_next_streak_for_test(
        int  current_streak,
        bool valid_draft);

bool common_dflash_invalid_reduced_logits_fail_closed_for_test(
        int invalid_streak,
        int threshold);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

// fork: per-slot init (n_seq=1, shared drafter context)
llama_context * common_speculative_create_ctx_dft(const common_params_speculative & params, int dflash_n_slots = 1);
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_dft_shared = nullptr);

// fork: single-seq overloads
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// fork: DFlash slot routing
void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id);

// fork: single-seq draft (returns tokens)
llama_tokens common_speculative_draft(
        common_speculative              * spec,
        const common_params_speculative & params,
        const llama_tokens              & prompt_tgt,
        llama_token                       id_last,
        std::vector<float>              * draft_log_probs = nullptr,
        llama_pos                         n_past_override = -1);

// fork: batched multi-slot DFlash drafting
void common_speculative_draft_batch(
        std::vector<common_speculative *> & specs,
        llama_context                     * ctx_dft,
        const common_params_speculative   & params,
        const std::vector<llama_token>    & id_last_per_spec,
        std::vector<llama_tokens>         & result_per_spec,
        std::vector<std::vector<float>>   * log_probs_per_spec = nullptr);

// fork: logit/state management
void   common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted);
void   common_speculative_update_logits_deferred_dflash_kv(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted);
void   common_speculative_update_logits_by_indices(common_speculative * spec, llama_context * ctx, const std::vector<int> & capture_indices);
int    common_speculative_flush_prefill(common_speculative * spec, int src_offset = 0, int n_tokens = 0, int commit_end_pos = 0);
void   common_speculative_set_prefill_capture_enabled(common_speculative * spec, bool enabled);
void   common_speculative_discard_dflash_state(common_speculative * spec, const char * reason);
void   common_speculative_note_prefill_suffix_scheduled(common_speculative * spec);

void   common_speculative_rollback_dft(common_speculative * spec, llama_seq_id seq_id, llama_pos n_past, uint16_t n_accepted);

// fork: DFlash ring buffer state save/load
size_t common_speculative_ring_state_size(const common_speculative * spec);
bool   common_speculative_ring_state_save(const common_speculative * spec, uint8_t * buf, size_t size);
bool   common_speculative_ring_state_load(common_speculative * spec, const uint8_t * buf, size_t size);
common_dflash_ring_stats common_speculative_dflash_ring_stats(const common_speculative * spec);

common_speculative_tree common_speculative_draft_tree(
        common_speculative              * spec,
        const common_params_speculative & params,
        const llama_tokens              & prompt_tgt,
        llama_token                       id_last,
        int                               tree_budget);

// fork: draft length params
int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);
