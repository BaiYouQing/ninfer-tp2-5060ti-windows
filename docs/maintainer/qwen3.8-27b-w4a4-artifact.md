# Qwen3.8-27B W4A4 NVFP4 artifact

This page defines how to build the `qwen3.8-27b/nvfp4-w4a4` `.ninfer` artifact from a W4A4 NVFP4
source checkpoint, how to verify an existing artifact against its source, and the fused row-order
conventions the converter implements. The generic `.ninfer` framing is defined in
[artifact-container.md](artifact-container.md), the numeric codecs in
[tensor-formats.md](tensor-formats.md), and byte packing in
[storage-layouts.md](storage-layouts.md). The registered `qwen3.8-27b/nvfp4` and
`qwen3.8-27b/groupwise-int` contracts are defined in
[qwen3.8-27b-artifact.md](qwen3.8-27b-artifact.md).

## 1. Identity

```text
filename   = qwen3_8_27b_nvfp4w4a4.ninfer
model_id   = qwen3.8-27b
weights_id = nvfp4-w4a4
```

`nvfp4-w4a4` is a distinct registered identity: it reuses the Qwen3.6-27B NVFP4 tier (its recipe and
inventory) but keeps the nine layers that tier leaves in BF16 as NVFP4, because the W4A4 source stores
them as NVFP4. The engine resolves the identity without a runtime profile flag.

## 2. Source checkpoint

The W4A4 source is a self-contained ModelOpt NVFP4 checkpoint (a merged Qwen3.8-27B fine-tune whose
linear layers are NVFP4: 4-bit weights with 4-bit activation quantization; control, head, MTP, and
vision modules retained in BF16). It is not redistributed by this repository; a reader regenerates the
`.ninfer` artifact from the source with the converter below.

| Fact | Value |
|---|---|
| source | [`nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4`](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4/tree/main/W4A4), subdirectory `W4A4` (`variant = fast`) |
| producer | ModelOpt NVFP4, `quant_algo = NVFP4`, `group_size = 16` |
| layout | one indexed checkpoint: `model-nvfp4-fast.safetensors` (18,822,252,240 B; NVFP4 text) + `vision-mtp-bf16.safetensors` (1,770,897,648 B; BF16 vision/MTP) + `model.safetensors.index.json` + the six frontend resources |
| binding verification | `source_package_sha256` in the source `manifest.json`, plus the per-file `SHA256SUMS` |
| source manifest facts | `precision = "W4A4 NVFP4 with retained BF16 control/head modules"`, `text_linear_counts = {NVFP4: 400, FP8: 0, BF16: 97}`, `language_model_layers = 64`, `tensor_count = 2399`, `file_bytes = 20616312752` |

Verify the downloaded source before converting:

```bash
cd /path/to/w4a4-source
sha256sum -c SHA256SUMS
```

and confirm that `manifest.json` reports `source_package_sha256 = 2d2eac20ceb1439ab85eda4c5d616150f1c1f4956729333cc6e6e15e49739b21`.

## 3. Conversion

The converter lives in `tools/convert/qwen3_8_27b/convert_w4a4.py` and reuses
`tools/convert/common/` for I/O and the engine's own encoders for quantization:

```bash
cd ninfer-tp2-5060ti
python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
  --src /path/to/w4a4-source \
  --out out/qwen3_8_27b_nvfp4w4a4.ninfer
```

The result is one complete image: 1310 tensor objects plus the six frontend resources. The object
inventory is the Qwen3.6-27B NVFP4 inventory with the nine exception layers restored to NVFP4 and
their `input_scale_divisor` objects added.

The `--draft-ranking` argument selects the frequency corpus used to derive the optimized draft head;
it defaults to the repository's `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`.

## 4. Verification

`--verify` recomputes every object with the same code path the converter uses and byte-compares it
against the stored payload, in addition to a contract check (name/format/layout/shape) and a
frontend-resource byte comparison:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
  --src /path/to/w4a4-source \
  --verify out/qwen3_8_27b_nvfp4w4a4.ninfer
```

A pass prints `VERIFY_DONE` and exits 0. NVFP4 objects are repacked byte-exactly, so a byte match is
the strongest available proof that the artifact is faithful to the source; the W8/Q4/Q5/Q6 endpoint,
MTP, and vision objects are re-quantized through the engine's own encoder and compared byte-for-byte.

## 5. Fused row order and known limitations

The fused projection objects carry a fixed physical row order that the engine's tensor-parallel
leaves expect. The converter implements these conventions:

| object | source | physical row order |
|---|---|---|
| `attention/query_key_gate_value` | `q_proj` + `k_proj` + `v_proj` | `[Q \| K \| Gate \| V]`, where the query and gate halves are taken per-head from `q_proj` (each head is 512 rows: query 256 + gate 256) |
| `mlp/gate_up` | `gate_proj` + `up_proj` | `[gate \| up]` |
| `gdn/query_key_value_z` | `in_proj_qkv` + `in_proj_z` | `[qkv \| z]` |

Known limitations and caveats:

- **The nine exception layers are NVFP4, not BF16.** The Qwen3.6-27B NVFP4 recipe leaves six attention
  `query_key_gate_value` layers (3/7/11/15/19/23), two attention `output` layers (3/7), and the GDN
  `output` layer (4) in BF16; the W4A4 source stores them as NVFP4, so the converter keeps them that
  way instead of down-converting. This states a property of the source, not of any artifact's
  tensor-parallel support: upstream's `qwen3.8-27b/nvfp4` artifact (whose exception layers are
  row-scaled FP8, not BF16) is the artifact the TP2 campaign in the repository README was run on.
  Whether this W4A4 tier and that FP8 tier behave identically under `--tp 2` on this fork's hardware
  has not been measured side by side.
- **`gdn/convolution` is transposed, not reshaped.** The source stores the GDN causal-convolution
  weight as `(C, 1, K)`; the engine reads it as `[K, C]`. The converter transposes (and asserts the
  shape). A plain reshape would keep the flat order and silently corrupt all 48 GDN layers.
- **NVFP4 nibble order is low-nibble-first.** `dequant_nvfp4` assumes the ModelOpt/FP4 packing
  convention (low nibble first). If a future source changes the packing, the dequantization table and
  the repack must be updated together.
- **The optimized draft head is derived, not stored in the source.** It is computed from the output
  head plus the `--draft-ranking` frequency corpus, so a different corpus yields a different draft
  head (and a different `text/draft_head_token_ids`). The source MTP module is the base-model MTP
  (not retrained with this fine-tune), so speculative acceptance is lower than a retrained MTP would
  give.