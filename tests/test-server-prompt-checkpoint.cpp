#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>

int main() {
    server_prompt_checkpoint ckpt {
        /*.pos_min  = */ 1,
        /*.pos_max  = */ 2,
        /*.n_tokens = */ 3,
        /*.data     = */ std::vector<uint8_t>(128),
    };
    ckpt.ring_data.resize(256);

    ckpt.clear();

    assert(ckpt.pos_min == 0);
    assert(ckpt.pos_max == 0);
    assert(ckpt.n_tokens == 0);
    assert(ckpt.data.empty());
    assert(ckpt.data.capacity() == 0);
    assert(ckpt.ring_data.empty());
    assert(ckpt.ring_data.capacity() == 0);

    server_prompt prompt;
    for (int i = 0; i < 3; ++i) {
        auto & cur = prompt.checkpoints.emplace_back();
        cur.n_tokens = i + 1;
        cur.data_tgt.resize((size_t) (i + 1) * 100);
    }

    {
        server_prompt budgeted = server_prompt_clone_with_checkpoint_budget(prompt, 250, 600);
        assert(budgeted.checkpoints.size() == 1);
        assert(budgeted.checkpoints.front().n_tokens == 3);
        assert(server_prompt_checkpoints_size(budgeted.checkpoints) == 300);
    }

    {
        server_prompt budgeted = server_prompt_clone_with_checkpoint_budget(prompt, 600, 600);
        assert(budgeted.checkpoints.empty());
    }

    {
        server_prompt budgeted = server_prompt_clone_with_checkpoint_budget(prompt, 0, 0);
        assert(budgeted.checkpoints.size() == 3);
        assert(server_prompt_checkpoints_size(budgeted.checkpoints) == 600);
    }

    prompt.data.main.resize(64);
    prompt.data.drft.resize(32);
    prompt.clear();
    assert(prompt.n_tokens() == 0);
    assert(prompt.data.size() == 0);
    assert(prompt.checkpoints.empty());

    return 0;
}
