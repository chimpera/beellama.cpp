# Quickstart: Gemma 4 31B with DFlash on a Single GPU

Run Gemma 4 31B IT with DFlash speculative decoding in [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) on one NVIDIA GPU. This guide covers model download, binary setup, and a launch command tuned for a 24 GB VRAM card (RTX 3090, RTX 4090, A5000, etc.).

DFlash is a speculative decoding mode where a small draft model reads recent hidden states from the target model and predicts multiple tokens ahead. The target model then verifies those predictions in a single forward pass. When the draft model is good, this produces multiple accepted tokens per target evaluation.

## Prerequisites

**Hardware.** An NVIDIA GPU with at least 24 GB VRAM. AMD GPUs on ROCm and Apple Silicon on macOS also work with limitations (see [Platform notes](#platform-notes)).

**Software.** One of:

- Windows or Linux: a prebuilt CUDA 12.4 or 13.1 binary, or build from source with CUDA Toolkit and CMake.
- macOS: a prebuilt Apple Silicon Metal binary, or build from source with Xcode command-line tools and CMake. DFlash runs on the CPU ring path only.

## Get the binary

### Prebuilt

Current release binaries are on the [releases page](https://github.com/Anbeeld/beellama.cpp/releases):

| Platform | Backend | Archive |
| --- | --- | --- |
| macOS arm64 | Metal | `bin-macos-arm64.tar.gz` |
| Ubuntu x64 | CPU | `bin-ubuntu-x64.tar.gz` |
| Ubuntu arm64 | CPU | `bin-ubuntu-arm64.tar.gz` |
| Ubuntu x64 | CUDA 12.4 | `bin-ubuntu-cuda-12.4-x64.tar.gz` |
| Ubuntu x64 | CUDA 13.1 | `bin-ubuntu-cuda-13.1-x64.tar.gz` |
| Ubuntu x64 | Vulkan | `bin-ubuntu-vulkan-x64.tar.gz` |
| Ubuntu x64 | ROCm 7.2 | `bin-ubuntu-rocm-7.2-x64.tar.gz` |
| Ubuntu x64 | SYCL | `bin-ubuntu-sycl-x64.tar.gz` |
| Windows x64 | CPU | `bin-win-cpu-x64.zip` |
| Windows x64 | Vulkan | `bin-win-vulkan-x64.zip` |
| Windows x64 | SYCL | `bin-win-sycl-x64.zip` |
| Windows x64 | CUDA 12.4 | `bin-win-cuda-12.4-x64.zip` |
| Windows x64 | CUDA 13.1 | `bin-win-cuda-13.1-x64.zip` |
| Windows x64 | HIP/Radeon | `bin-win-hip-radeon-x64.zip` |

Windows CUDA archives contain a `ggml-cuda.dll` backend; download the matching `cudart-win-cuda-*-x64.zip` runtime archive and extract it into the same folder. Windows Vulkan requires a Vulkan-capable GPU driver/runtime on the target machine. Windows SYCL and HIP archives ship as standalone packages with their required runtime DLLs bundled, except for GPU driver runtimes.

Docker images are published to `ghcr.io/anbeeld/beellama.cpp`:

| Image | Acceleration | Platforms |
| --- | --- | --- |
| `server`, `server-cpu` | CPU | linux/amd64, linux/arm64 |
| `server-cuda`, `server-cuda12` | CUDA 12.4 | linux/amd64 |
| `server-cuda13` | CUDA 13.1 | linux/amd64 |
| `server-rocm` | ROCm | linux/amd64 |
| `server-vulkan` | Vulkan | linux/amd64 |
| `server-sycl` | SYCL | linux/amd64 |

Building from source with `-DGGML_NATIVE=ON` may result in a tiny bit better performance, so it can still be worth doing if you plan to use this fork long-term.

### Build from source

**Windows (MSVC + CUDA).** Run in PowerShell or Command Prompt (CMake finds MSVC automatically; no Developer Command Prompt needed):

```powershell
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

**Linux (GCC + CUDA).**

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`GGML_CUDA_FA_ALL_QUANTS=ON` is required for TurboQuant and TCQ cache types. Add `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 3090, or `-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 4090, if cross-compiling or building in CI without a GPU.

**macOS (Metal).**

```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build produces `llama-server` (or `llama-server.exe` on Windows) inside `build/bin/` or `build/bin/Release/`.

## Get the models

You need three files: a target model, a DFlash draft model, and a multimodal projector (mmproj). The mmproj is optional; skip it and remove the `--mmproj` / `--no-mmproj-offload` flags if you do not need vision.

Three practical combos make sense right now. Pick one based on whether you want the best overall 24 GB setup, a higher-quality target with tighter VRAM margins, or a lighter setup with more headroom.

### Model configurations

| Combo | Target | Drafter | K cache | V cache | Best for |
|-------|--------|---------|---------|---------|----------|
| **Balanced** | Q4_K_S | IQ4_XS or Q4_K_M or Q5_K_M | q5_0 | q4_1 | Current default on 24 GB cards; strongest overall quality/speed/VRAM trade from the live launcher config |
| **Precision** | UD-Q4_K_XL | IQ4_XS or Q4_K_M or Q5_K_M | q5_0 | q4_1 | Higher target quality when you can afford it; limited VRAM headroom for context on 24 GB cards |
| **VRAM** | IQ4_XS or Q3_K_M or Q3_K_S | IQ4_XS or Q4_K_M | q4_0 | q4_0 (or turbo3_tcq for tighter VRAM) | Lower VRAM, more context headroom, or heavier background VRAM pressure |

The precision combo gives the best model quality. The `q5_0` K cache plus `q4_1` V cache is the current default, as my recent long-context KV quant benchmarks showed it is a better quality/size trade than the older TurboQuant recommendation.

As for heavier drafters like Q8, from my testing it was always a net negative for tok/s, probably due to combination of the model being much larger, and quantization not mattering as much for drafting as all the model needs to do is guess 8-16 tokens at a time. But if someone will provide benchmarks that prove otherwise or point out it's due to an incorrect implementation in the code, I'll be happy to change the stance.

### Where to download

**Target model** - from [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)

**DFlash draft model** - from [Anbeeld/gemma-4-31B-it-DFlash-GGUF](https://huggingface.co/Anbeeld/gemma-4-31B-it-DFlash-GGUF)

The draft model shares the target's token embedding and LM head at runtime, so the GGUF file only contains the DFlash-specific weights.

### Multimodal projector (optional)

Download from [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) (look for `mmproj-BF16.gguf`).

## Launch

Replace the model paths with your own.

**If your machine has unified memory (Mac, Ryzen APU, etc.), do not use `--no-mmproj-offload`** - there is no separate VRAM pool to save.

### Balanced combo (current default)

Q4_K_S target, IQ4_XS or Q4_K_M or Q5_K_M drafter, `q5_0` K cache, `q4_1` V cache. Vision is enabled but is offloaded to CPU as thus consumes no VRAM, which seems to work with no issues whatsoever thanks to fixes implemented in the fork.

Note that this config has 100k context (`--ctx-size 102400`) and `-ub 512` (important for prefill speed) by default. These values are considered safe and should work even with Windows reserving some VRAM and a couple of Electron-based VRAM hogs being open at the same time.

I can personally confirm that on my RTX 3090 24 GB after fresh Windows 11 restart and with minimum background apps, I was able to launch the balanced combo with **140k context** and chat with the model about some stuff through **WebUI in Chromium**, with 2 monitors connected to dGPU and HWiNFO 64 open on the second monitor for checking stats.

This of course consumed 99.5% of my VRAM, but after killing the server the system was still using **700MB VRAM**. Which basically means that if you eliminate VRAM pressure from other sources, you should be very comfortable with the balanced combo above 100k context on a single 3090 or 4090.

So if you are not on Windows, or your monitors are connected to iGPU freeing up gGPU VRAM, or your monitors are turned off and all apps are closed, etc. etc. you can freely experiment with higher values up to like 200k context, or look towards higher quality models like UD-Q4_K_XL.

Also there's `--spec-dflash-cross-ctx 1024` which is how much context the drafter sees at the time. Higher values eat up more VRAM, but in my testing they didn't increase tok/s as this slowed down the drafter itself quite a bit. Default value is `512`, but for longer context `1024` seemed better from testing, still might be something worth tinkering with.

**Windows (PowerShell):**

```powershell
llama-server.exe `
  -m "path\to\gemma-4-31B-it-Q4_K_S.gguf" `
  --mmproj "path\to\mmproj-BF16.gguf" `
  --no-mmproj-offload `
  --spec-draft-model "path\to\gemma4-31b-it-dflash-IQ4_XS.gguf" `
  --spec-type dflash `
  --spec-dflash-cross-ctx 1024 `
  --port 8082 `
  -np 1 `
  --kv-unified `
  -ngl all `
  --spec-draft-ngl all `
  -b 2048 -ub 512 `
  --ctx-size 102400 `
  --cache-type-k q5_0 --cache-type-v q4_1 `
  --flash-attn on `
  --jinja `
  --no-mmap --mlock `
  --no-host `
  --reasoning on `
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0
```

**Linux / macOS:**

```bash
llama-server \
  -m "path/to/gemma-4-31B-it-Q4_K_S.gguf" \
  --mmproj "path/to/mmproj-BF16.gguf" \
  --no-mmproj-offload \
  --spec-draft-model "path/to/gemma4-31b-it-dflash-IQ4_XS.gguf" \
  --spec-type dflash \
  --spec-dflash-cross-ctx 1024 \
  --port 8082 \
  -np 1 \
  --kv-unified \
  -ngl all \
  --spec-draft-ngl all \
  -b 2048 -ub 512 \
  --ctx-size 102400 \
  --cache-type-k q5_0 --cache-type-v q4_1 \
  --flash-attn on \
  --jinja \
  --no-mmap --mlock \
  --no-host \
  --reasoning on \
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0
```

Unlike the Qwen config, the current Gemma launcher does not set `--chat-template-kwargs '{"preserve_thinking":true}'`. The command above matches the actual Gemma launcher state.

### Precision combo (higher-quality target)

This uses `UD-Q4_K_XL` instead of `Q4_K_S` while keeping the rest of the balanced command the same.

On a 24 GB card this leaves limited VRAM headroom for context, so treat it as a tighter fit than the balanced combo. If it does not fit comfortably at `--ctx-size 102400`, reduce context first.

Change these flags from the balanced command above:

```text
-m "path\to\gemma-4-31B-it-UD-Q4_K_XL.gguf"
```

### Speed / VRAM combo (lighter)

Use a lighter target and lighter cache. A practical starting point is:

```text
-m "path\to\gemma-4-31B-it-IQ4_XS.gguf"
--spec-draft-model "path\to\gemma4-31b-it-dflash-IQ4_XS.gguf"
--cache-type-k q4_0
--cache-type-v q4_0
```

If you need even more KV headroom on CUDA, try:

```text
--cache-type-k turbo3_tcq
--cache-type-v q4_1
--reasoning off
--ctx-size 50000
```

That mirrors the current `Start-Gemma4-31B-DFlash-NoReason.ps1` launcher, which trades some context behavior and quality for lower VRAM pressure.

### Optionally: use `--spec-draft-hf` instead of a local file

Replace `--spec-draft-model "path/to/draft.gguf"` with:

```text
--spec-draft-hf Anbeeld/gemma-4-31B-it-DFlash-GGUF:IQ4_XS
```

This downloads the draft model from Hugging Face on first run and caches it locally.

## What the flags do

For full command-line tuning, including upstream llama.cpp args, DFlash args, TurboQuant/TCQ cache choices, context checkpoints, prompt-cache RAM, and chat/reasoning controls, read [beellama-args.md](beellama-args.md).

### DFlash flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `--spec-type` | `dflash` | Enables DFlash speculative decoding |
| `--spec-draft-model` | path or HF repo | DFlash draft model to load |
| `--spec-draft-ngl` | `all` | Offload all draft layers to GPU |
| `--spec-dflash-cross-ctx` | `1024` | How many tokens of target hidden state the drafter sees. Higher gives more context to cross-attention, lower saves VRAM |

### Context and cache flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `--ctx-size` | `102400` | Total KV context allocation. Lower to save VRAM |
| `-b` | `2048` | Logical batch size for prompt evaluation |
| `-ub` | `512` | Physical microbatch size |
| `--kv-unified` | - | Single KV buffer shared across server slots |
| `-ngl` | `all` | Offload all target model layers to GPU |
| `--cache-type-k` | `q5_0` | K cache quantization. `q5_0` is the current default for the precision Gemma setup |
| `--cache-type-v` | `q4_1` | V cache quantization. `q4_1` keeps the cache footprint reasonable while preserving better tail behavior |
| `--flash-attn` | `on` | Use Flash Attention kernels |
| `--jinja` | - | Enable Jinja template engine for chat formatting |

### Model and sampling flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `--mmproj` | path | Multimodal projector for vision input |
| `--no-mmproj-offload` | - | Run mmproj on CPU, freeing GPU VRAM at a latency cost (skip on macOS - unified memory has no separate VRAM pool) |
| `--reasoning` | `on` | Enable reasoning output handling |
| `--temp` | `1.0` | Sampling temperature |
| `--top-k` | `64` | Top-K sampling |
| `--top-p` | `0.95` | Top-P sampling |
| `--min-p` | `0.0` | Min-P sampling (0 = disabled) |

### Server and infrastructure flags

| Flag | Value | What it controls |
|------|-------|-----------------|
| `-np` | `1` | Parallel slots (DFlash works with one slot by default) |
| `--port` | `8082` | HTTP listen port |
| `--no-host` | - | Bypass host buffer, allowing extra buffers to be used |
| `--no-mmap` | - | Load model into memory instead of memory-mapping |
| `--mlock` | - | Lock model pages in RAM to prevent swapping |

## Platform notes

### NVIDIA CUDA (Windows, Linux)

Full DFlash acceleration: GPU cross-attention ring buffer, device-to-device hidden-state capture and replay, GPU tape path for Gemma 4. All TurboQuant and TCQ cache types are available.

### Apple Metal (macOS)

DFlash runs through the CPU ring buffer path - functional but slower than CUDA - because there is no GPU cross-attention ring on Metal. The recommended `q5_0` / `q4_1` cache works as the normal path. Only `turbo3` and `turbo4` are available from the TurboQuant family on Metal; `turbo2` and the TCQ types (`turbo2_tcq`, `turbo3_tcq`) are CUDA-only.

### AMD ROCm

DFlash falls back to the CPU ring buffer path. Standard cache types such as `q5_0` and `q4_1` are the recommended starting point. The ROCm build compiles TurboQuant from the same CUDA source files (via HIP), so TurboQuant and TCQ cache types may work, but compilation success under HIPCC is not guaranteed. If TCQ types fail, stay on standard cache types or try non-TCQ TurboQuant. Build with `-DGGML_HIP=ON` instead of `-DGGML_CUDA=ON`.

### Vulkan

Not recommended for DFlash. Falls back to CPU ring with no TurboQuant cache types.

### Multi-GPU

Tree verification (`--spec-branch-budget` > 0) is automatically disabled when the target model spans more than one GPU, but it is still very slow and is not part of any recommended config here. Flat DFlash (`--spec-branch-budget 0`) still works across multiple GPUs. Set `GGML_DFLASH_GPU_RING=0` to disable the GPU ring buffer for isolation or debugging on multi-GPU setups.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_DFLASH_GPU_RING` | enabled | Set to `0` to disable the GPU cross-attention ring buffer and force CPU-only ring |
| `GGML_DFLASH_MAX_CTX` | `4096` | Cap cross-attention context length in tokens. Set to `0` for unlimited |
| `GGML_DFLASH_PROFILE` | `0` | `1` / `default` enables summary, replay, copy, and verify timing. Add categories such as `prefill` or `trace` for deeper profiling |
| `GGML_DFLASH_DEBUG` | `0` | Enable DFlash debug logs such as prefill route and capture decisions |
| `GGML_DFLASH_CRASH_TRACE` | `0` | Enable high-volume crash breadcrumbs around recurrent backup and decode sync points |
| `GGML_DFLASH_KV_CACHE_MODE` | `both` | DFlash K/V cache mode: `off`/`none`/`disabled` disables, `k`/`k-only` keeps K only, `v`/`v-only` keeps V only, unset or any other value keeps both |

## Adjusting for your hardware

The default command targets 24 GB VRAM with `Q4_K_S`. If you are running out of memory, adjust in this order:

1. **Reduce `--ctx-size`.** Each unit of context costs VRAM for both the target model's KV cache and the DFlash cross-attention buffer. Dropping from `102400` to `65536` or `32768` frees significant memory.
2. **Switch cache types.** Replace `q5_0` / `q4_1` with `q4_0` / `q4_0` first. On CUDA, `turbo3_tcq` for both K and V squeezes further; `turbo2_tcq` is the last resort. On Metal, use `turbo3` for both K and V if `q4_0` is too large.
3. **Drop the target quantization.** Move from `Q4_K_S` to `IQ4_XS`, then to `Q3_K_M` or `Q3_K_S` if needed.
4. **Reduce `--spec-dflash-cross-ctx`.** Lowering from `1024` to `512` saves VRAM at the cost of less context for the drafter's cross-attention.
5. **Lower `--cache-ram`.** The prompt-cache RAM subsystem defaults to 8192 MiB. If system RAM is tight, lower it to 4096 or 2048 to reduce the RAM ceiling:

   ```
   --cache-ram 4096
   ```

6. **Remove `--mlock`.** If system RAM is abundant and swapping is not a concern, `--mlock` can be removed.

Start with the balanced combo and drop down if VRAM is tight.

### When you have VRAM to spare

If you have spare VRAM after fitting the balanced combo, spend it on:

- **Higher `--ctx-size`** - push past 100K context without hitting the ceiling.
- **Heavier cache types** - try `q5_0` / `q5_0` if you value precision over memory savings.
- **Heavier target quants** - if you have locally built or trusted higher-quality Gemma GGUFs that still fit, move the target up before making the drafter heavier.

## Adaptive draft depth

By default, the server adjusts draft depth using the `profit` controller, which raises and lowers the active draft depth (up to the ceiling set by `--spec-draft-n-max`) based on real-time acceptance rates. You do not need to change anything to benefit from this.

Adaptive draft is highly configurable, so if you want to tinker with it, check out [beellama-args.md](beellama-args.md). For fixed-depth benchmarking, use `--no-spec-dm-adaptive --spec-draft-n-max N`.

## Troubleshooting

**Out of VRAM.** Reduce `--ctx-size` first, then cache types, then target quantization. See [Adjusting for your hardware](#adjusting-for-your-hardware).

**`spec-type dflash is set but draft model is not a DFlash drafter`.** Bee accepts two DFlash drafter GGUF schemas: `dflash-draft` for the Bee/buun schema, and `dflash` for the upstream llama.cpp DFlash PR schema. If loading fails, check the exact error for missing DFlash metadata keys or tensors. A plain Gemma model is still not a DFlash drafter.

**Port already in use.** Change `--port` or stop the existing server.

**Slow DFlash on macOS.** The CPU ring path is slower than the CUDA GPU ring. This is a platform limitation, not a configuration issue. Reducing `--spec-dflash-cross-ctx` to `512` lowers CPU ring overhead.

**TCQ cache types fail on non-CUDA backends.** `turbo2_tcq` and `turbo3_tcq` are CUDA-only. Use standard cache types such as `q5_0`, `q4_0`, `q8_0`, or `f16` instead. On Metal, non-TCQ TurboQuant is limited to `turbo3` and `turbo4`; on ROCm, non-TCQ TurboQuant may work if the HIP build succeeds.

**DFlash seems disabled.** Check the server log for `dflash:` or `speculative` lines. If DFlash is active, you will see draft acceptance rates and timing. If you see no DFlash output, verify that `--spec-type dflash` is set and the draft model loaded successfully. A DFlash draft GGUF auto-detects as `dflash` even without `--spec-type`, but setting it explicitly avoids ambiguity.
