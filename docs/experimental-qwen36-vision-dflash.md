# Experimental Qwen3.6 vision + DFlash

This branch extends native Qwen3.6 `mmproj` support with opt-in DFlash decode
after multimodal prefill. Autoregressive decoding remains the default.

## Provenance

The native vision implementation comes directly from upstream pull request
[Luce-Org/lucebox#571](https://github.com/Luce-Org/lucebox/pull/571),
`feat(vision): native mmproj multimodal chat for Qwen35`, authored by
[David Roth](https://github.com/davidmroth).

This branch was created from the PR head commit:

- source repository: `davidmroth/lucebox-hub`
- source branch: `feat/vision-native-mmproj`
- commit: `122393626cf7bc488e6a27d2ea7069ff15316312`
- upstream base: `Luce-Org/lucebox:main`

The PR provides `--mmproj`, OpenAI-compatible image input parsing, mtmd image
encoding, multimodal Qwen35 prefill, and an AR safety fallback for image
requests. This fork preserves that implementation and adds:

1. opt-in DFlash decode for single-GPU multimodal requests;
2. correct selection of the final prefill argmax/logits row;
3. correct logits offset when the final text chunk exceeds the prefill
   micro-batch size.

## Enabling vision DFlash

Build with native vision enabled:

```bash
cmake -S server -B server/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DDFLASH27B_SERVER=ON \
  -DDFLASH27B_MMPROJ=ON
cmake --build server/build --target dflash_server -j
```

PR #571 requires a full `llama.cpp`/mtmd dependency. The tested build used
`Luce-Org/lucebox-ggml` branch `luce-dflash`, with this branch's patched
`server/deps/llama.cpp/ggml` subtree overlaid onto it before configuring CMake.

Run the experimental mode:

```bash
DFLASH_VISION_DFLASH=1 DFLASH27B_PREFILL_UBATCH=256 \
server/build/dflash_server Qwen3.6-27B-Q4_K_M.gguf \
  --draft dflash-draft-3.6-q4_k_m.gguf \
  --mmproj mmproj-Qwen3.6-27B-Q8_0.gguf \
  --ddtree --ddtree-budget 22 \
  --chunk 256 --max-ctx 61440 \
  --cache-type-k q4_0 --cache-type-v q4_0
```

Without `DFLASH_VISION_DFLASH=1`, image requests use the PR's AR behavior.
Text-only requests continue to use DFlash normally. Layer-split/multi-GPU
vision remains AR-only because its committed-position handling is different
and was not validated by this experiment.

## RTX 3090 notes

The tested target was a 24 GB RTX 3090. A BF16 projector left insufficient
headroom for a roughly 690 MiB multimodal prefill allocation after a 5K-token
prompt. The official `ggml-org/Qwen3.6-27B-GGUF`
`mmproj-Qwen3.6-27B-Q8_0.gguf` reduced projector residency enough to complete
the same request while keeping GPU vision prefill fast.

The tested llama-swap profile used:

- 61,440-token context;
- Q4_0 K/V cache;
- prefill chunk and micro-batch size 256;
- Q8_0 mmproj on GPU;
- DDTree budget 22.

## Limitations

This is an experiment, not a claim of production-ready image acceleration.
Observed image-request acceptance was often only 10-17%, so end-to-end decode
speed stayed close to AR. A deterministic image test also produced a semantic
divergence from AR. Use the default AR path when visual fidelity matters and
enable vision DFlash only for evaluation.

DFlash accelerates text decoding after image encoding. It does not accelerate
the vision encoder or multimodal prefill.
