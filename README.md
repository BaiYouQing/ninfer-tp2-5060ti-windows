# NInfer

**English** | [简体中文](README.zh-CN.md)

> Tensor-parallel NInfer on two consumer cards. Qualified on **2× RTX 5060 Ti (16 GiB each)**: one
> 27B model resident across both GPUs, a **253,952-token single-slot context**, four KV-cache tiers
> (`bf16` / `int8` / `fp8` / `k16v8`), MTP3 speculative decoding with prefix reuse that actually
> hits, and a `/health` that reports engine availability.

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen checkpoints. It runs
text, image, and video prompts through a local CLI or OpenAI-/Anthropic-compatible HTTP APIs, on one
GPU, or -- for the 27B execution package -- tensor-parallel across two: see
[Dual-GPU (TP2)](#dual-gpu-tp2). The paragraphs below describe upstream; what this fork adds and
measures is in the fork note.

> **This is a fork.** Upstream is [Neroued/ninfer](https://github.com/Neroued/ninfer); this tree
> branches from its commit `feaf4dd` via the TP2 line
> ([wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m),
> [giocom/ninfer-3060X2](https://github.com/giocom/ninfer-3060X2)) and, on that base, adds **dual-GPU
> tensor parallelism** (`--tp 2 --devices A,B`) to the 27B execution package: one resident model, one
> process, two devices, no NVLink and no distributed serving, halving per-card weight and KV residency.
> This fork further adds **KV-cache tiers** (`--kv-dtype bf16|int8|fp8|k16v8`), a **working MTP prefix
> reuse at `--tp 2`**, and a **truthful `/health`** with supervisor-driven self-heal; those three were
> measured on **2× RTX 5060 Ti (16 GiB each)**. `--tp 1` output is byte-identical to `feaf4dd` on the
> greedy cases in [`tests/data/tp1-golden/`](tests/data/tp1-golden/MANIFEST.md), and single-GPU
> behaviour, supported identities, artifact format, and protocol surfaces are unchanged. The design
> decisions, numerical contracts, and qualification evidence are in
> [Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md) -- that document is
> upstream's, and also covers the YaRN extended-position work this fork does not use.
> See [NOTICE](NOTICE) for attribution.

## The weights

This fork runs one model: **Qwen3.8-27B NVFP4 W4A4**. The `.ninfer` artifact is not distributed here;
build it from the published source with the in-tree converter.

| | |
|---|---|
| Source | [`nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4`](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4/tree/main/W4A4), subdirectory `W4A4` — ModelOpt NVFP4, group size 16, `variant = fast` |
| Source files | `model-nvfp4-fast.safetensors` (18,822,252,240 B, SHA-256 `9b7e1c4d839995ee9ed35ac682ecf31e81ae4bc6ddb4e3aaa7f585b786bb83a0`) and `vision-mtp-bf16.safetensors` (1,770,897,648 B), plus the index and six frontend resources |
| Source check | the directory's `SHA256SUMS`, and `manifest.json` with `source_package_sha256 = 2d2eac20ceb1439ab85eda4c5d616150f1c1f4956729333cc6e6e15e49739b21` |
| Converter | [`tools/convert/qwen3_8_27b/convert_w4a4.py`](tools/convert/qwen3_8_27b/convert_w4a4.py) |
| Result | `qwen3_8_27b_nvfp4w4a4.ninfer`, 17,555,334,916 bytes (16.35 GiB), SHA-256 `63c204d223e73d63d6d4db8a82aa3f4859592cd83b00545bcee38334643341cb` |

### How the artifact is built

```bash
# 1. the W4A4 subdirectory of the published checkpoint
hf download nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4 \
  --include "W4A4/*" --local-dir src

# 2. convert
python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
  --src src/W4A4 --out models/qwen3_8_27b_nvfp4w4a4.ninfer

# 3. verify: recompute every object and byte-compare it against the artifact
python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
  --src src/W4A4 --verify models/qwen3_8_27b_nvfp4w4a4.ninfer
```

`huggingface.co` is not reachable from every network. `huggingface_hub` honours the `HF_ENDPOINT`
environment variable, so `HF_ENDPOINT=https://hf-mirror.com hf download …` fetches the same files
through a mirror — that is how these steps were run when this fork's conversion was last reproduced.

The converter reads nine files from `src/W4A4`: the two safetensors shards, `model.safetensors.index.json`,
and six frontend resources (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`). The six
frontend resources are copied into the artifact byte-for-byte and the conversion aborts if any is
missing, so the `W4A4/*` glob above is the right thing to download — it fetches everything the
converter needs, at roughly 13 MB beyond the two shards.

What the converter does, and what it deliberately does not do:

- it reuses the engine's own encoders. NVFP4 objects are **repacked byte-exactly**, BF16/FP32 objects
  pass through, and the W8/Q4/Q5/Q6 endpoint, the MTP module and the vision tower are re-quantized
  through the same encoder the engine loads with;
- it emits the fused projections in the physical row order the tensor-parallel shards expect:
  `attention/query_key_gate_value` as `[Q | K | Gate | V]` (query and gate taken per head from
  `q_proj`, 256 + 256 rows per head), `mlp/gate_up` as `[gate | up]`, and `gdn/query_key_value_z` as
  `[qkv | z]`;
- it **transposes** `gdn/convolution`: the source stores it `(C, 1, K)` and the engine reads
  `[K, C]`, so a plain reshape would keep the flat order and silently corrupt all GDN layers;
- it keeps nine layers as NVFP4 that the Qwen3.6 NVFP4 recipe leaves in BF16. The source already
  stores them as NVFP4, so the converter preserves the source instead of down-converting it;
- it builds the optimized proposal head from a frequency ranking (`--draft-ranking`, default
  `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`). The ranking holds, per token, how
  often the model's own teacher-forced argmax emitted it over a 72k-conversation corpus; the
  converter takes the top 131,072 tokens (of 248,320) plus every special token, slices those rows out
  of the output head into `text/draft_head`, and writes their ids as `text/draft_head_token_ids`. The
  head is computed, not copied out of the source, so a different corpus yields a different head.
  Both the algorithm and the corpus are **upstream's**: the selection and materialization live in
  `tools/convert/qwen3_6/common/draft_head.py`, and the ranking fixtures ship in `tools/freq_corpus/`
  with upstream's tree. This tier only wires them to the W4A4 source.

The verification gate is a byte comparison, and the conversion this fork ships passes it on **all
1310 tensor objects plus the six frontend resources**. The three steps above were re-run from the
published source on this fork's machine and produced a file **byte-identical** to the artifact it
ships (same SHA-256 as the table above). The full page is
[docs/maintainer/qwen3.8-27b-w4a4-artifact.md](docs/maintainer/qwen3.8-27b-w4a4-artifact.md).

### Other registered artifacts

NInfer deliberately supports a closed set of artifacts rather than acting as a general model runtime.
The engine also registers upstream's identities — the official
[`Qwen3.8-27B NVFP4`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) artifact
(`qwen3_8_27b_nvfp4.ninfer`, 21,492,695,040 bytes, SHA-256
`bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32`), Qwen3.6-27B in both weight
profiles, Qwen3.8-27B `groupwise-int`, and Qwen3.6-35B-A3B — and accepts them; they are outside what
this fork is built and measured against, with one measurement worth recording: the official `nvfp4`
artifact was also run at `--tp 2` here, and it loads and generates normally on 2× RTX 5060 Ti (10.08
GiB of weights per card; 38 tok/s decode with MTP off, and 76–86 tok/s with `--spec mtp
--draft-tokens 3 --lm-head-draft`). The W4A4 form carries 16% less weight per card and decodes at
106 tok/s on the same `int8` tier. With `int8` KV, a single slot fits that
artifact's whole 262,144-token native ceiling and still leaves 842 MiB free per card — the same
residual under `ninfer-serve` as under the CLI, because the media and response buffers are not
reserved while vision is off. The W4A4 form
is the model this fork wanted to run, not a tensor-parallel requirement. Current builds accept only
the version-2 container, and all of those are version 2.

Every `.ninfer` file contains the weights and frontend resources NInfer needs. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file. Each artifact is complete, while GPU
residency is fixed at process startup. Speculative decoding is off by default, so MTP/DFlash state and
the optimized proposal head are not uploaded; vision is off by default too, so its weights, the Vision
scratch phase and the frozen request-transient allocation are omitted. Add `--vision` to the CLI or
server process that must accept image or video input, and `--spec mtp --draft-tokens 3` to the one
that must speculate. Disabled capabilities cannot be enabled by a later request.

## Quick start

Clone **this** repository (not upstream, and not the TP2 forks it descends from):

```bash
git clone https://github.com/lynx-gt/ninfer-tp2-5060ti.git
cd ninfer-tp2-5060ti

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Serve the artifact built in [The weights](#the-weights) on two GPUs with the K16V8 KV cache
(253,952-token single slot, MTP3 speculative decoding with the optimized draft head):

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4w4a4.ninfer \
  --host 0.0.0.0 --port 8815 --model-id qwen3.8-27b-w4a4-mtp3 \
  --tp 2 --devices 0,1 --kv-dtype k16v8 \
  --max-context 253952 --kv-capacity 253952 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --lm-head-draft --max-concurrency 1 --cors
```

[Requirements](#requirements), [Build](#build) and
[Dual-GPU (TP2)](#dual-gpu-tp2) cover the prerequisites, artifact conversion and the complete option
set.

## Performance

This README carries this fork's own numbers in
[KV-cache tiers and long-context limits](#kv-cache-tiers-and-long-context-limits): 2× RTX 5060 Ti,
`--tp 2`, one slot, per-tier prefill/decode/acceptance and the long-context ceilings.

Upstream's single-GPU RTX 5090 campaign (Qwen3.6-27B in both weight profiles, Qwen3.6-35B-A3B, and
the Qwen3.8-27B NVFP4 artifact) is documented in [Performance](docs/performance.md) and in the model
cards. Those figures were not measured here and are not comparable to figures taken on two 16 GiB
cards.

## Evaluation

This fork has no budget for a capability evaluation of its own, so no benchmark table is published
here. For scores, see the model repositories: the per-artifact model cards under
[`model-cards/`](model-cards/) and the artifacts' Hugging Face pages carry upstream's results
(AIME 2025/2026, GPQA-Diamond, ERQA, RealWorldQA, EvalScope 1.9.0, single sample). Those were
measured on upstream's artifacts, not on this fork's conversion.

## Requirements

NInfer currently requires:

- 64-bit Linux;
- one NVIDIA GeForce RTX 5090 (`sm_120a`), or two for `--tp 2`;
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

The build rejects CUDA architectures other than `120a`. There is no install target or packaged
binary distribution; NInfer is run from its source build tree.

## Build

Clone **this** repository — upstream has no `--tp 2`, and neither do the other TP2 forks carry these
KV tiers.

```bash
git clone https://github.com/lynx-gt/ninfer-tp2-5060ti.git
cd ninfer-tp2-5060ti

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Dual-GPU (TP2)

`--tp 2` splits one resident model across two GPUs: one process, one resident model, two CUDA
devices, no NVLink and no distributed serving. It is a capacity feature rather than scale-out — it
halves per-card weight and KV residency. It is implemented for the 27B execution package
(`qwen3.6-27b` and `qwen3.8-27b`, either weight profile); `qwen3.6-35b-a3b` has no tensor-parallel
path and rejects `--tp 2` at startup.

Every two-card number this README publishes was taken on **2× RTX 5060 Ti (16 GiB each)** against the
W4A4 artifact described under [The weights](#the-weights).

### Usage

```bash
# both GPUs, K16V8 KV, a 253,952-token single slot, MTP3 with the optimized proposal head
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4w4a4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 253952 --kv-capacity 253952 --prefill-chunk 1024 \
  --kv-dtype k16v8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-concurrency 1
```

The same flags drive the CLI; add `--no-thinking` when a short `--max-new` budget has to reach the
answer channel, because at this checkpoint's default thinking mode a 16-token budget is consumed
entirely inside the reasoning stream:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4w4a4.ninfer \
  --tp 2 --devices 0,1 \
  --max-context 253952 --kv-capacity 253952 --kv-dtype k16v8 \
  --messages long_prompt.json --max-new 256 --no-thinking
```

- `--tp 2` requires an explicit `--devices A,B` naming two distinct devices of the same compute
  capability. `--tp 1` remains the default and is unchanged: `scripts/tp1-regression.sh` compares
  this build's greedy output against token streams recorded from upstream `ninfer` at base commit
  `feaf4dd`, over a short chat, a 2,191-token instruction and a 28,677-token document, and requires
  the token ids, the generated text and the deterministic summary rows to be byte-equal. Its scope
  is exactly that: greedy text decode, `--tp 1`, `--rope native`, NVFP4 weights, `qwen3.8-27b`, MTP
  off, concurrency 1. See `tests/data/tp1-golden/MANIFEST.md`.
- `--kv-capacity` must be at least `--max-context`. The explicit form is what fits a tier at its
  ceiling; `auto` also keeps a 512 MiB sizing headroom.
- `--max-concurrency 1` is arithmetic on 16 GiB cards: the tier table below is what one slot costs,
  and `k16v8` at 253,952 leaves no room for a second.
- MTP speculative decoding (`--spec mtp --draft-tokens 1..5`, optionally `--lm-head-draft`) works at
  `--tp 2`, including compatible-prefix reuse. `--spec dflash` and `--vision` are rejected at
  `--tp 2`.
- `--rope yarn`, the extended-position path this line inherits from upstream, is available but unused
  here: this fork's ceiling is 253,952 tokens, below the registered native 262,144, so no rope
  scaling is needed and none is measured.

See the [CLI guide](docs/cli.md) and [HTTP serving](docs/serving.md) for the full option contract.

### KV-cache tiers and long-context limits

`--kv-dtype` selects the KV codec per side: `bf16` (no quantization), `int8` (per-64 fp16 scale),
`fp8` (e4m3, per-256 fp16 scale), `k16v8` (BF16 keys + FP8 values). All tiers compute QK in BF16;
the codec only changes residency and read-back.

Measured on 2× RTX 5060 Ti (TP2, one slot, 8k-token prompt, `--spec mtp --draft-tokens 3
--lm-head-draft --prefill-chunk 1024`):

| `--kv-dtype` | prefill | decode | MTP acceptance | 1-slot context ceiling |
|---|---|---|---|---|
| `int8` | 4880 tok/s | 106.5 tok/s | 3.04 tok/round | 262144 |
| `fp8` | 4170 tok/s | 91.4 tok/s | 2.69 tok/round | 262144 |
| `k16v8` | 4480 tok/s | 88.2 tok/s | 2.56 tok/round | 253952 (chunk 1024) / 229376 (chunk 4096) |
| `bf16` | — | — | 2.89 tok/round | — |

Per-round cost is the same across tiers (28.5–29.4 ms); the token-rate spread comes from the
acceptance length the speculative pairing reaches. `--kv-capacity` must be ≥ `--max-context`, and
the explicit-capacity form is what fits a tier at its ceiling (`auto` keeps a 512 MiB sizing
headroom, which `k16v8` cannot afford at 253952).

Long context (k16v8, 1 slot, 253952): a 57.7k-token prompt prefills in 30.4 s (1905 tok/s), a
192.6k-token prompt in 287 s (672 tok/s) with a correct mid-context needle and constant 15.7 GiB
per GPU — prefill is O(T²) and gets slow, decode is unaffected (82–96 tok/s).

Prefix reuse hits when the retained turn checkpoint covers the prompt: a repeated request reports
`cache=7872 reuse=restore_turn_checkpoint` and its time-to-first-token drops from 1788 ms to 71 ms.
A reused prefill computes its suffix from the checkpoint frontier rather than from the full-prefill
chunk grid, so a hit and a cold run can differ in the last bits (and occasionally in greedy text) —
the same class of caveat as vLLM/SGLang prefix caching. The hit is reported back in the response
usage: OpenAI responses carry `usage.prompt_tokens_details.cached_tokens`, and Anthropic responses
carry `usage.cache_read_input_tokens` (with `cache_creation_input_tokens` reported as 0).

`/health` reports engine availability (`200 {"status":"ok"}` / `503 {"status":"unavailable"}`), and
`apps/ninfer-serve` exits non-zero when the engine becomes unusable so a supervisor
(`Restart=on-failure`) reloads the model in about 16 s instead of leaving a dead endpoint up.

### Limitations

- **Vision is `--tp 1` only.** The Vision encoder runs on the primary device against replicated
  weights and has no split path, so `--tp 2 --vision` is rejected at startup.
- **DFlash is rejected at `--tp 2`.** It remains a 35B-A3B text-only backend, and that target has no
  tensor-parallel path at all.
- **Peer access depends on the board.** Upstream measured `cudaDeviceCanAccessPeer` as 0 between two
  RTX 5090s and stages the collectives as host-staged copies over PCIe rather than direct peer
  copies; on the two RTX 5060 Ti cards this fork was measured on, it reports 1 in both directions.
  Either way a decode token costs 128 reduces plus one logit all-gather, and under CUDA Graphs the
  whole collective set is small next to the ~28 ms per round the weights themselves cost.
- **MTP prefix reuse works at `--tp 2`, with two known degradations.** This fork implements the
  retained-state resume on the TP2 path, so a request that extends a prefix the engine still holds
  hits (`reuse=restore_turn_checkpoint`) instead of re-prefilling the whole prompt. It still
  degrades to a full prefill when the retained base already covers the entire prompt, so that there
  is no suffix left to compute; and a hit is a *same-prefix continuation*, not a shared prefix --
  a request whose common prefix is not the one the engine retained does not hit. The answer is
  unchanged in every case and no request fails.
- **MTP is output-equivalent up to near-tie argmax flips, not bit-identical.** A verify round
  evaluates the target model over `K+1` columns at once and an ordinary round over one, which
  selects different GEMM shapes; greedy MTP-on and MTP-off streams can therefore diverge on a
  near-tie token. Every committed token is still one the target model's own argmax selected.
- **`--ignore-eos` is a diagnostic flag.** It exists for fixed-length soak and throughput work.
  Generation past the end-of-turn token is off-distribution and is not a product output. It is not
  the only route to a degenerate stream: greedy decoding at long context also loops on its own
  content, with the flag off.
- **A reused prefix can differ from a cold prefill in the last bits.** A cache hit computes its
  suffix from the retained checkpoint frontier instead of from the full-prefill chunk grid, so the
  two runs are not bit-identical and greedy text can occasionally flip. This is the same class of
  caveat as vLLM/SGLang prefix caching, and it is why a cache hit is not a correctness contract.
- **The FP8 tiers prefill about 9–14% slower than `int8`.** The FP8 KV staging path is not yet
  asynchronous; decode is unaffected.
- **`k16v8` cannot reach 262144 tokens in a single slot.** Its BF16 keys cost 26.2 KiB/token, so the
  one-slot ceiling is 253952 with `--prefill-chunk 1024` (and 229376 with `--prefill-chunk 4096`).

The design decisions behind the TP2 path -- the collective transport, the shard map, and what each
correctness gate actually proves -- are in
[Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md); that document also
covers the YaRN extended-position work this fork does not use.

## Capabilities and limits

Capabilities. All three registered model IDs (`qwen3.6-27b`, `qwen3.8-27b`, `qwen3.6-35b-a3b`)
support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- KV cache tiers `bf16`, `int8` (group-64), `fp8` (e4m3) and `k16v8` (BF16 keys + FP8 values);
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse, including MTP at `--tp 2`, with the hit count reported as
  `usage.prompt_tokens_details.cached_tokens` (OpenAI) / `usage.cache_read_input_tokens`
  (Anthropic);
- a `/health` endpoint that reports engine availability and a server that exits non-zero when the
  engine becomes unusable, so a supervisor can restart it;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen.

Limits.

- Only the five registered `(model_id, weights_id)` artifact identities are accepted product
  identities; this fork is built and measured against the Qwen3.8-27B NVFP4 pair listed above.
- Execution is specialized for the RTX 5090. One CUDA device is the default; the 27B execution
  package also runs on exactly two with `--tp 2 --devices A,B`, which is a capacity feature rather
  than scale-out. This fork adds measurements on 2× RTX 5060 Ti (16 GiB); the build target is
  `sm_120a` either way.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  CPU/GPU offload, or distributed serving. Multi-GPU execution is exactly the two-device
  tensor-parallel width described above: one process, one resident model, no NVLink, no more than
  two devices.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' native 262,144-token limit (`--rope yarn` reaches further on the 27B targets, but this fork
  does not use it; see [Dual-GPU (TP2)](#dual-gpu-tp2)). `--kv-capacity N` explicitly sizes the shared Main Text KV
  pool for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Relationship to upstream

This fork descends from `Neroued/ninfer` at commit `feaf4dd` (2026-08-20) through the TP2 line, and
carries its own work on that base. Upstream `master` has since advanced more than 200 commits and
restructured its runtime -- its executor and KV sizing live in different files than here, and its
KV-cache subsystem was rewritten. The two trees are therefore **not** interchangeable: merging
`master` into this line conflicts in hundreds of files, so this fork tracks its own base and
cherry-picks upstream fixes where they apply instead of following `master`. GitHub reports this
branch as ahead of *and* behind `Neroued:master` -- that is the state of this line, not staleness.

| | this fork | upstream `master` |
|---|---|---|
| `--kv-dtype` | `bf16`, `int8`, `fp8`, **`k16v8`** (BF16 keys + FP8 values) | `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4` |
| Tensor parallelism | `--tp 2 --devices A,B`, validated on 2× RTX 5090 and 2× RTX 5060 Ti | single GPU |
| `/health` | engine availability, plus a supervisor-driven restart when the engine dies | engine availability |

If you want `nvfp4` / `k8v4` KV tiers or upstream's newest single-GPU scheduling work, use upstream.
If you want tensor-parallel serving on two consumer cards with the `k16v8` tier and MTP prefix reuse
that actually hits, this fork is the line to use.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). This fork's modifications are under the
same licence; see [NOTICE](NOTICE) for the attribution required by Apache-2.0 §4(b).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
