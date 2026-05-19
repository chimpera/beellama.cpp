#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static bool expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
    }
    return ok;
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string mtmd_h = read_file(root + "/tools/mtmd/mtmd.h");
    const std::string clip_h = read_file(root + "/tools/mtmd/clip.h");
    const std::string mtmd_cpp = read_file(root + "/tools/mtmd/mtmd.cpp");
    const std::string clip_cpp = read_file(root + "/tools/mtmd/clip.cpp");
    const std::string mtmd_helper = read_file(root + "/tools/mtmd/mtmd-helper.cpp");
    const std::string server_context = read_file(root + "/tools/server/server-context.cpp");
    const std::string mtmd_cli = read_file(root + "/tools/mtmd/mtmd-cli.cpp");
    const std::string mtmd_debug = read_file(root + "/tools/mtmd/debug/mtmd-debug.cpp");

    ok &= expect(mtmd_h.find("int decoder_n_ubatch") != std::string::npos,
        "mtmd context params must carry the decoder physical ubatch");
    ok &= expect(clip_h.find("int decoder_n_ubatch") != std::string::npos,
        "clip context params must receive the decoder physical ubatch");
    ok &= expect(mtmd_cpp.find("/* decoder_n_ubatch */ 0") != std::string::npos,
        "mtmd default params must leave decoder ubatch unspecified for API callers without a decoder context");

    ok &= expect(server_context.find("mparams.decoder_n_ubatch = llama_n_ubatch(ctx)") != std::string::npos,
        "server mmproj setup must pass the text decoder ubatch to mtmd");
    ok &= expect(mtmd_cli.find("mparams.decoder_n_ubatch = llama_n_ubatch(lctx)") != std::string::npos,
        "mtmd CLI setup must pass the text decoder ubatch to mtmd");
    ok &= expect(mtmd_debug.find("mparams.decoder_n_ubatch = llama_n_ubatch(llama_init->context())") != std::string::npos,
        "mtmd debug setup must pass the text decoder ubatch to mtmd");

    ok &= expect(clip_cpp.find("clip_set_limit_image_tokens_for_non_causal_decode(hparams, 252, 280, decoder_n_ubatch)") != std::string::npos,
        "Gemma4 image token limits must be capped to decoder ubatch for non-causal image decode");
    ok &= expect(clip_cpp.find("loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION, ctx_params.decoder_n_ubatch)") != std::string::npos,
        "clip initialization must pass decoder ubatch into vision hparam loading");
    ok &= expect(mtmd_cpp.find("/* decoder_n_ubatch */ ctx_params.decoder_n_ubatch") != std::string::npos,
        "mtmd must forward decoder ubatch to clip initialization");

    ok &= expect(mtmd_helper.find("const int32_t n_ubatch = llama_n_ubatch(lctx)") != std::string::npos,
        "mtmd image decode must inspect the decoder physical ubatch");
    ok &= expect(mtmd_helper.find("non-causal attention requires the full") != std::string::npos,
        "mtmd image decode must fail gracefully when a non-causal chunk exceeds ubatch");
    ok &= expect(mtmd_helper.find("lower --image-max-tokens") != std::string::npos,
        "mtmd image decode error must tell users how to avoid the non-causal ubatch limit");
    return ok ? 0 : 1;
}
