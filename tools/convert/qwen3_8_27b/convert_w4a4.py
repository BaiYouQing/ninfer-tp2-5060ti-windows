"""Build the W4A4 (NVFP4 W4A4) Qwen3.8-27B ``.ninfer`` artifact.

The W4A4 source is a self-contained ModelOpt NVFP4 checkpoint: a merged
Qwen3.8-27B fine-tune whose linear layers are NVFP4 (4-bit weights with
4-bit activation quantization) and whose control, head, MTP, and vision
modules are retained in BF16.

This converter reuses the registered Qwen3.6-27B NVFP4 tier (its recipe and
inventory) and, on top of that, keeps the nine layers the Q36 tier leaves in
BF16 as NVFP4 (the source already stores them as NVFP4), adding the matching
``input_scale_divisor`` objects.  No other format conversion is performed:
NVFP4 objects are repacked byte-exactly, BF16/FP32 objects pass through, and
the W8/Q4/Q5/Q6 endpoint, MTP, and vision objects are quantized through the
engine's own encoder.

The fused projection objects carry a fixed physical row order that the engine
expects (see ``docs/maintainer/qwen3.8-27b-w4a4-artifact.md``):

* ``attention/query_key_gate_value``  ->  ``[Q | K | Gate | V]`` where the
  query and gate halves are taken per-head from the source ``q_proj``;
* ``mlp/gate_up``                     ->  ``[gate | up]``;
* ``gdn/query_key_value_z``           ->  ``[qkv | z]``.

Canonical invocation (convert)::

    python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
      --src /path/to/w4a4-source \
      --out out/qwen3_8_27b_nvfp4w4a4.ninfer

Verify an existing artifact against its source (recompute every object and
byte-compare it against the stored payload)::

    python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
      --src /path/to/w4a4-source \
      --verify out/qwen3_8_27b_nvfp4w4a4.ninfer
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from safetensors import safe_open

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)
from tools.artifact.layouts import encode_direct, encode_nvfp4
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_6_27b import inventory_nvfp4 as q36_inventory
from tools.convert.qwen3_6_27b import recipe_nvfp4 as q36_recipe

MODEL_ID = "qwen3.8-27b"
DEFAULT_WEIGHTS_ID = "nvfp4-w4a4"
_NVFP4_LAYOUT = "blockscale-k16-m128x4-v1"

# The nine layers the Qwen3.6 NVFP4 tier leaves in BF16.  The W4A4 source
# stores them as NVFP4, so this tier restores them to NVFP4.
_EXC_QKV = {3, 7, 11, 15, 19, 23}
_EXC_OUT = {3, 7}
_EXC_GDN = {4}

_FRONTEND_RESOURCES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",
    "frontend/generation_config.json",
    "frontend/preprocessor_config.json",
    "frontend/video_preprocessor_config.json",
)

# ---------------------------------------------------------------- source -> object name map
P = "model.language_model.layers.{L}."
LAYER_DIRECT = {
    "input_norm": "input_layernorm.weight",
    "post_attention_norm": "post_attention_layernorm.weight",
    "attention/query_norm": "self_attn.q_norm.weight",
    "attention/key_norm": "self_attn.k_norm.weight",
    "attention/output": "self_attn.o_proj.weight",
    "gdn/a_log": "linear_attn.A_log",
    "gdn/dt_bias": "linear_attn.dt_bias",
    "gdn/convolution": "linear_attn.conv1d.weight",
    "gdn/a_projection": "linear_attn.in_proj_a.weight",
    "gdn/b_projection": "linear_attn.in_proj_b.weight",
    "gdn/a_b_projection": "linear_attn.in_proj_a.weight",
    "gdn/norm": "linear_attn.norm.weight",
    "gdn/output": "linear_attn.out_proj.weight",
    "mlp/down": "mlp.down_proj.weight",
}
VISION_LAYER = {
    "attention/qkv": "attn.qkv.weight",
    "attention/qkv_bias": "attn.qkv.bias",
    "attention/output": "attn.proj.weight",
    "attention/output_bias": "attn.proj.bias",
    "mlp/fc1": "mlp.linear_fc1.weight",
    "mlp/fc1_bias": "mlp.linear_fc1.bias",
    "mlp/fc2": "mlp.linear_fc2.weight",
    "mlp/fc2_bias": "mlp.linear_fc2.bias",
    "norm1/weight": "norm1.weight",
    "norm1/bias": "norm1.bias",
    "norm2/weight": "norm2.weight",
    "norm2/bias": "norm2.bias",
}
MTP_DIRECT = {
    "mtp/input_projection": "mtp.fc.weight",
    "mtp/embedding_norm": "mtp.pre_fc_norm_embedding.weight",
    "mtp/hidden_norm": "mtp.pre_fc_norm_hidden.weight",
    "mtp/final_norm": "mtp.norm.weight",
    "mtp/layer/input_norm": "mtp.layers.0.input_layernorm.weight",
    "mtp/layer/post_attention_norm": "mtp.layers.0.post_attention_layernorm.weight",
    "mtp/layer/attention/query_norm": "mtp.layers.0.self_attn.q_norm.weight",
    "mtp/layer/attention/key_norm": "mtp.layers.0.self_attn.k_norm.weight",
    "mtp/layer/attention/output": "mtp.layers.0.self_attn.o_proj.weight",
    "mtp/layer/mlp/down": "mtp.layers.0.mlp.down_proj.weight",
}

_DIVISOR_TO_OBJECT: dict[str, str] = {}
_DRAFT_CACHE: dict[str, object] = {}


def _stem(name: str) -> str:
    return name[: -len(".weight")] if name.endswith(".weight") else name


def sources_for(name: str) -> list[str]:
    """Map one artifact object to the source safetensors tensor(s) it comes from."""
    if name.endswith("/input_scale_divisor"):
        parent = _DIVISOR_TO_OBJECT.get(name)
        if parent is None:
            raise KeyError(f"unknown divisor object: {name}")
        return [_stem(m) + ".input_scale" for m in sources_for(parent)]
    if name == "text/token_embedding":
        return ["model.language_model.embed_tokens.weight"]
    if name == "text/output_head":
        return ["lm_head.weight"]
    if name == "text/final_norm":
        return ["model.language_model.norm.weight"]
    if name in ("text/draft_head", "text/draft_head_token_ids"):
        return ["lm_head.weight"]
    if name.startswith("mtp/"):
        if name == "mtp/layer/attention/query_key_gate_value":
            b = "mtp.layers.0.self_attn."
            return [b + "q_proj.weight", b + "k_proj.weight", b + "v_proj.weight"]
        if name == "mtp/layer/mlp/gate_up":
            b = "mtp.layers.0.mlp."
            return [b + "gate_proj.weight", b + "up_proj.weight"]
        return [MTP_DIRECT[name]]
    if name.startswith("vision/layers/"):
        parts = name.split("/")
        return [f"model.visual.blocks.{parts[2]}.{VISION_LAYER['/'.join(parts[3:])]}"]
    if name == "vision/patch_embedding":
        return ["model.visual.patch_embed.proj.weight"]
    if name == "vision/patch_embedding_bias":
        return ["model.visual.patch_embed.proj.bias"]
    if name == "vision/position_embedding":
        return ["model.visual.pos_embed.weight"]
    if name.startswith("vision/merger/"):
        table = {
            "fc1": "model.visual.merger.linear_fc1.weight",
            "fc1_bias": "model.visual.merger.linear_fc1.bias",
            "fc2": "model.visual.merger.linear_fc2.weight",
            "fc2_bias": "model.visual.merger.linear_fc2.bias",
            "norm/weight": "model.visual.merger.norm.weight",
            "norm/bias": "model.visual.merger.norm.bias",
        }
        return [table[name[len("vision/merger/"):]]]
    if name.startswith("text/layers/"):
        parts = name.split("/")
        layer, suffix = int(parts[2]), "/".join(parts[3:])
        b = P.format(L=layer)
        if suffix == "attention/query_key_gate_value":
            return [b + "self_attn.q_proj.weight", b + "self_attn.k_proj.weight",
                    b + "self_attn.v_proj.weight"]
        if suffix == "gdn/query_key_value_z":
            return [b + "linear_attn.in_proj_qkv.weight", b + "linear_attn.in_proj_z.weight"]
        if suffix == "mlp/gate_up":
            return [b + "mlp.gate_proj.weight", b + "mlp.up_proj.weight"]
        if suffix == "gdn/a_b_projection":
            return [b + "linear_attn.in_proj_a.weight", b + "linear_attn.in_proj_b.weight"]
        if suffix in LAYER_DIRECT:
            return [b + LAYER_DIRECT[suffix]]
    raise KeyError(f"no mapping rule for: {name}")


def divisor_name_for(object_name: str) -> str:
    """Name the ``input_scale_divisor`` object that pairs with an NVFP4 object."""
    prefix, suffix = object_name.rsplit("/", 1)
    if suffix == "query_key_gate_value" and prefix.endswith("/attention"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/attention"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "query_key_value_z" and prefix.endswith("/gdn"):
        return prefix + "/input_projection/input_scale_divisor"
    if suffix == "output" and prefix.endswith("/gdn"):
        return prefix + "/output_projection/input_scale_divisor"
    if suffix == "gate_up" and prefix.endswith("/mlp"):
        return prefix + "/gate_up_projection/input_scale_divisor"
    if suffix == "down" and prefix.endswith("/mlp"):
        return prefix + "/down_projection/input_scale_divisor"
    raise KeyError(f"no input_scale_divisor rule for: {object_name}")


def _force_nvfp4(name: str) -> bool:
    parts = name.split("/")
    if len(parts) >= 4 and parts[0] == "text" and parts[1] == "layers":
        layer = int(parts[2])
        suffix = "/".join(parts[3:])
        if suffix == "attention/query_key_gate_value" and layer in _EXC_QKV:
            return True
        if suffix == "attention/output" and layer in _EXC_OUT:
            return True
        if suffix == "gdn/output" and layer in _EXC_GDN:
            return True
    return False


def build_w4a4_specs() -> tuple[list[tuple[str, tuple, str, str]], dict[str, str]]:
    """Derive the W4A4 tensor specs: the Q36 tier with its nine BF16 exception
    layers restored to NVFP4 and the matching divisor objects added."""
    base = [(sp.name, tuple(sp.shape), sp.format, sp.layout) for sp in q36_inventory.TENSOR_SPECS]
    flipped: list[str] = []
    specs: list[tuple[str, tuple, str, str]] = []
    for name, shape, fmt, layout in base:
        if _force_nvfp4(name):
            fmt, layout = "NVFP4", _NVFP4_LAYOUT
            flipped.append(name)
        specs.append((name, shape, fmt, layout))
    for name in flipped:
        div = divisor_name_for(name)
        if not any(item[0] == div for item in specs):
            specs.append((div, (), "FP32", "contiguous-le-v1"))
    divisor_to_object: dict[str, str] = {}
    for name, _shape, fmt, _layout in specs:
        if fmt == "NVFP4":
            divisor_to_object[divisor_name_for(name)] = name
    return specs, divisor_to_object


# ---------------------------------------------------------------- source reader
class SourceReader:
    """Read the W4A4 source: the indexed shards plus any unindexed shards."""

    def __init__(self, root: Path):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        self.indexed = ShardReader(self.root) if index.exists() else None
        covered = set(self.indexed.weight_map.values()) if self.indexed else set()
        self.extra: dict[str, Path] = {}
        for shard in sorted(self.root.glob("*.safetensors")):
            if shard.name in covered:
                continue
            with safe_open(str(shard), framework="pt") as handle:
                for name in handle.keys():
                    self.extra[name] = shard

    def has(self, name: str) -> bool:
        if self.indexed is not None and self.indexed.has(name):
            return True
        return name in self.extra

    def get(self, name: str) -> torch.Tensor:
        if self.indexed is not None and self.indexed.has(name):
            return self.indexed.get(name)
        if name in self.extra:
            with safe_open(str(self.extra[name]), framework="pt") as handle:
                return handle.get_tensor(name)
        raise KeyError(name)


class Adapter:
    """Resolve the recipe's source names against the ModelOpt W4A4 checkpoint.

    ModelOpt stores ``weight`` / ``weight_scale_2`` / ``input_scale``; the
    recipe expects ``weight_packed`` / ``weight_global_scale`` /
    ``input_global_scale``.  The two scale fields are multipliers in the source
    and divisors in the recipe, so they are inverted here.
    """

    _SUFFIX = {
        "weight_packed": "weight",
        "weight_global_scale": "weight_scale_2",
        "input_global_scale": "input_scale",
    }
    _INVERT = ("weight_global_scale", "input_global_scale")

    def __init__(self, src: SourceReader):
        self.src = src

    def get(self, name: str) -> torch.Tensor:
        base, _, suffix = name.rpartition(".")
        if suffix in self._SUFFIX:
            tensor = self.src.get(base + "." + self._SUFFIX[suffix])
            if suffix in self._INVERT:
                return 1.0 / tensor.to(torch.float32)
            return tensor
        return self.src.get(name)


# ---------------------------------------------------------------- encoders
def dequant_nvfp4(codes: torch.Tensor, scales_u8: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """NVFP4 dequantization: E2M1 codes (low nibble first) x block scale (E4M3)
    x global divisor -> BF16."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
    n, k2 = codes.shape
    lo = (codes & 0x0F).to(torch.long)
    hi = ((codes >> 4) & 0x0F).to(torch.long)
    values = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(n, k2 * 2)
    scales = scales_u8.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=1)
    return (values * scales * divisor.to(torch.float32).reshape(())) .to(torch.bfloat16)


def _get_draft_context(src_dir: Path, ranking_path: Path):
    key = str(ranking_path)
    if key not in _DRAFT_CACHE:
        _DRAFT_CACHE[key] = draft_head.compute_shortlist(ranking_path, src_dir)
    return _DRAFT_CACHE[key]


def encode_object(
    name: str,
    shape: tuple,
    fmt: str,
    mem: list[str],
    reader: SourceReader,
    src_dir: Path,
    spec_obj: TensorSpec,
    adapter: Adapter,
    ranking_path: Path,
) -> bytes:
    if name == "text/draft_head":
        context = _get_draft_context(src_dir, ranking_path)
        tensor = draft_head.materialize_draft_head(reader.get("lm_head.weight"), context)
        return family_conversion.encode_tensor_payload(tensor, spec_obj, "cpu")
    if name == "text/draft_head_token_ids":
        context = _get_draft_context(src_dir, ranking_path)
        return family_conversion.encode_tensor_payload(
            draft_head.materialize_draft_head_token_ids(context), spec_obj, "cpu"
        )

    if name.endswith("/input_scale_divisor"):
        sel = q36_recipe.INPUT_DIVISORS_BY_NAME.get(name)
        if sel is not None:
            return encode_direct(q36_recipe.materialize_input_divisor(sel, adapter), "FP32")
        values = [1.0 / reader.get(m).to(torch.float32).reshape(()) for m in mem]
        for value in values[1:]:
            if not torch.equal(value, values[0]):
                raise SystemExit(f"{name}: fused members disagree on input_scale")
        return encode_direct(values[0], "FP32")

    stems = [_stem(m) for m in mem]
    nvfp4_src = all(reader.has(s + ".weight_scale") for s in stems)

    if nvfp4_src:
        if fmt == "NVFP4":
            pre_sel = q36_recipe.NVFP4_WEIGHTS_BY_NAME.get(name)
            if pre_sel is not None:
                packed, scales, divisor = q36_recipe.materialize_nvfp4_weight(pre_sel, adapter)
                return encode_nvfp4(packed, scales, divisor, shape)
            if name.endswith("/attention/query_key_gate_value") and len(mem) == 3:
                q, k, v = mem
                q_c, k_c, v_c = reader.get(q), reader.get(k), reader.get(v)
                q_s = reader.get(_stem(q) + ".weight_scale").view(torch.uint8)
                k_s = reader.get(_stem(k) + ".weight_scale").view(torch.uint8)
                v_s = reader.get(_stem(v) + ".weight_scale").view(torch.uint8)
                idx_q = torch.cat([torch.arange(h * 512, h * 512 + 256) for h in range(24)])
                idx_g = torch.cat([torch.arange(h * 512 + 256, h * 512 + 512) for h in range(24)])
                packed = torch.cat([q_c[idx_q], k_c, q_c[idx_g], v_c], dim=0)
                scales = torch.cat([q_s[idx_q], k_s, q_s[idx_g], v_s], dim=0)
                divisor = 1.0 / reader.get(_stem(q) + ".weight_scale_2").to(torch.float32).reshape(())
                return encode_nvfp4(packed, scales, divisor, shape)
            codes = [reader.get(m) for m in mem]
            scales = [reader.get(st + ".weight_scale").view(torch.uint8) for st in stems]
            d0 = 1.0 / reader.get(stems[0] + ".weight_scale_2").to(torch.float32).reshape(())
            code_t = codes[0] if len(codes) == 1 else torch.cat(codes, dim=0)
            scale_t = scales[0] if len(scales) == 1 else torch.cat(scales, dim=0)
            return encode_nvfp4(code_t, scale_t, d0.numpy().tobytes(), shape)
        # target BF16/FP32: dequantize the NVFP4 source
        codes = [reader.get(m) for m in mem]
        scales = [reader.get(s + ".weight_scale").view(torch.uint8) for s in stems]
        d0 = reader.get(stems[0] + ".weight_scale_2").to(torch.float32).reshape(())
        code_t = codes[0] if len(codes) == 1 else torch.cat(codes, dim=0)
        scale_t = scales[0] if len(scales) == 1 else torch.cat(scales, dim=0)
        tensor = dequant_nvfp4(code_t, scale_t, d0)
        return encode_direct(tensor if fmt == "BF16" else tensor.to(torch.float32), fmt)

    # non-NVFP4 source (BF16/FP32 passthrough, or quantized to W8/Q4/Q5/Q6)
    tensor = reader.get(mem[0]) if len(mem) == 1 else torch.cat([reader.get(m) for m in mem], dim=0)
    if name.endswith("/attention/query_key_gate_value") and len(mem) == 3:
        q, k, v = (reader.get(m) for m in mem)
        heads = q.shape[0] // 512
        idx_q = torch.cat([torch.arange(h * 512, h * 512 + 256) for h in range(heads)])
        idx_g = torch.cat([torch.arange(h * 512 + 256, h * 512 + 512) for h in range(heads)])
        tensor = torch.cat([q[idx_q], k, q[idx_g], v], dim=0)
        if tuple(tensor.shape) != tuple(shape):
            raise SystemExit(f"{name}: shape {tuple(tensor.shape)} != spec {tuple(shape)} after row reorder")
    elif name.endswith("/gdn/convolution"):
        # transpose, not reshape: the source is (C, 1, K) and the engine reads [K, C]
        if tensor.dim() != 3 or tensor.shape[1] != 1:
            raise SystemExit(f"{name}: expected source (C,1,K), got {tuple(tensor.shape)}")
        c, _one, k = tensor.shape
        tensor = tensor.reshape(c, k).t().contiguous()
        if tuple(tensor.shape) != tuple(shape):
            raise SystemExit(f"{name}: shape {tuple(tensor.shape)} != spec {tuple(shape)} after transpose")
    elif len(shape) == 2 and tensor.dim() != 2:
        tensor = tensor.reshape(shape)
    if fmt in ("BF16", "FP32"):
        if fmt == "FP32" and tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return encode_direct(tensor, fmt)
    return family_conversion.encode_tensor_payload(tensor, spec_obj, "cpu")


# ---------------------------------------------------------------- resources
def load_resources(src_dir: Path) -> tuple[list[ResourceSpec], dict[str, bytes]]:
    res_specs: list[ResourceSpec] = []
    res_data: dict[str, bytes] = {}
    for resource_name in _FRONTEND_RESOURCES:
        path = src_dir / Path(resource_name).name
        if not path.exists():
            raise SystemExit(f"missing frontend resource: {path}")
        data = path.read_bytes()
        res_specs.append(ResourceSpec(name=resource_name, encoding="raw-bytes-v1", bytes=len(data)))
        res_data[resource_name] = data
    return res_specs, res_data


def _default_ranking() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "tools" / "freq_corpus" / "fixtures" / "ranking" / "ranking.train.counts.i64"


# ---------------------------------------------------------------- convert
def _write_all(
    out: Path,
    reader: SourceReader,
    src_dir: Path,
    res_specs: list[ResourceSpec],
    res_data: dict[str, bytes],
    specs: list[tuple[str, tuple, str, str]],
    by_name: dict[str, TensorSpec],
    adapter: Adapter,
    ranking_path: Path,
    weights_id: str,
    limit: int,
) -> int:
    identity = ArtifactIdentity(MODEL_ID, weights_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_specs = list(res_specs) + [
        TensorSpec(name=n, shape=tuple(s), format=f, layout=l) for n, s, f, l in specs
    ]
    writer = ArtifactWriter(out, identity, all_specs)
    stats: dict[str, int] = {}
    t0 = time.time()
    written = 0
    truncated = False
    try:
        for resource_name in _FRONTEND_RESOURCES:
            writer.write(resource_name, res_data[resource_name])
        for name, shape, fmt, _layout in specs:
            mem = sources_for(name)
            missing = [m for m in mem if not reader.has(m)]
            if missing and name not in ("text/draft_head", "text/draft_head_token_ids"):
                raise SystemExit(f"{name}: source missing tensors {missing}")
            payload = encode_object(name, shape, fmt, mem, reader, src_dir, by_name[name], adapter, ranking_path)
            writer.write(name, payload)
            stats[fmt] = stats.get(fmt, 0) + 1
            written += 1
            if written % 200 == 0:
                print(f"  {written}/{len(specs)}  {name}  ({time.time() - t0:.0f}s)", flush=True)
            if limit and written >= limit:
                truncated = True
                print(f"  reached --limit {limit} (artifact incomplete)")
                break
        if not truncated:
            writer.finish()
    finally:
        if truncated:
            writer.close()
    print(f"\ndone: {out}  {out.stat().st_size / 2**30:.2f} GiB  in {time.time() - t0:.0f}s")
    print("object stats:", {k: v for k, v in sorted(stats.items())})
    print(f"identity: {MODEL_ID} / {weights_id}")
    print("CONVERT_DONE")
    return 0


# ---------------------------------------------------------------- verify
def _verify(
    verify_path: Path,
    reader: SourceReader,
    src_dir: Path,
    res_data: dict[str, bytes],
    specs: list[tuple[str, tuple, str, str]],
    by_name: dict[str, TensorSpec],
    adapter: Adapter,
    ranking_path: Path,
) -> int:
    art = Artifact.open(verify_path)
    print(f"artifact identity: {art.identity.model_id} / {art.identity.weights_id}")
    if art.identity.model_id != MODEL_ID:
        print(f"  ! model_id mismatch (expected {MODEL_ID})")
    print(f"objects in file: {len(art.objects)}")
    objects = {o.name: o for o in art.objects}

    # 1. contract: names / formats / layouts / shapes vs the W4A4 spec
    expected = {n: (f, l, tuple(s)) for n, s, f, l in specs}
    missing, mismatch, extra = [], [], []
    for name, (fmt, layout, shape) in expected.items():
        obj = objects.get(name)
        if obj is None:
            missing.append(name)
            continue
        if (obj.format, obj.layout, tuple(obj.shape)) != (fmt, layout, shape):
            mismatch.append((name, (fmt, layout, shape), (obj.format, obj.layout, tuple(obj.shape))))
    for obj in art.objects:
        if obj.name not in expected and not obj.name.startswith("frontend/"):
            extra.append(obj.name)
    print(f"\n1. contract: {len(expected)} expected objects")
    print(f"   missing: {len(missing)}  mismatched: {len(mismatch)}  extra: {len(extra)}")
    for name, want, got in mismatch[:8]:
        print(f"   x {name}\n      expected {want}\n      actual   {got}")
    for name in missing[:5]:
        print(f"   x missing {name}")
    for name in extra[:5]:
        print(f"   ? extra {name}")

    # 2. frontend resources: byte-compare against the source directory
    print("\n2. frontend resources")
    for resource_name in _FRONTEND_RESOURCES:
        obj = objects.get(resource_name)
        if obj is None:
            print(f"   x missing {resource_name}")
            continue
        same = bytes(art.payload(obj)) == res_data[resource_name]
        print(f"   {'ok ' if same else 'BAD'} {resource_name}  {len(res_data[resource_name])} B  "
              f"{'matches source' if same else 'differs from source'}")

    # 3. per-object recompute: byte-compare against the engine's own encoders
    print("\n3. per-object recompute (byte-compare)")
    t0 = time.time()
    bad = 0
    checked = 0
    for name, shape, fmt, _layout in specs:
        mem = sources_for(name)
        if name not in ("text/draft_head", "text/draft_head_token_ids"):
            absent = [m for m in mem if not reader.has(m)]
            if absent:
                print(f"   BAD {name}: source missing {absent}")
                bad += 1
                continue
        want = encode_object(name, shape, fmt, mem, reader, src_dir, by_name[name], adapter, ranking_path)
        obj = objects.get(name)
        if obj is None:
            print(f"   BAD {name}: not in artifact")
            bad += 1
            continue
        got = bytes(art.payload(obj))
        if want != got:
            bad += 1
            if bad <= 8:
                first = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b),
                              min(len(want), len(got)))
                print(f"   BAD {name} [{fmt}]: recomputed {len(want)} B vs artifact {len(got)} B, "
                      f"first diff at byte {first}")
        checked += 1
        if checked % 200 == 0:
            print(f"   {checked}/{len(specs)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"   checked {checked} objects, {bad} differ  ({time.time() - t0:.0f}s)")

    ok = not missing and not mismatch and not extra and bad == 0
    print("\n=== verdict ===")
    if ok:
        print("PASS: contract, frontend, and every recomputed object byte-match the artifact.")
        print("VERIFY_DONE")
        return 0
    print("FAIL: see the mismatches above")
    print("VERIFY_FAILED")
    return 1


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="W4A4 source directory (safetensors + index + frontend resources)")
    parser.add_argument("--out", type=Path,
                        help="output .ninfer path (convert mode)")
    parser.add_argument("--verify", type=Path,
                        help="existing .ninfer to verify against the source (verify mode)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N tensor objects in convert mode (incomplete artifact)")
    parser.add_argument("--weights-id", default=DEFAULT_WEIGHTS_ID)
    parser.add_argument("--draft-ranking", type=Path, default=None,
                        help="frequency ranking fixture for the draft head "
                             "(default: the repository's ranking.train.counts.i64)")
    args = parser.parse_args()

    if bool(args.out) == bool(args.verify):
        parser.error("exactly one of --out (convert) or --verify (verify) is required")

    ranking_path = args.draft_ranking or _default_ranking()
    specs, divisor_to_object = build_w4a4_specs()
    _DIVISOR_TO_OBJECT.update(divisor_to_object)
    reader = SourceReader(args.src)
    adapter = Adapter(reader)
    res_specs, res_data = load_resources(args.src)
    by_name = {n: TensorSpec(name=n, shape=tuple(s), format=f, layout=l) for n, s, f, l in specs}

    if args.verify is not None:
        return _verify(args.verify, reader, args.src, res_data, specs, by_name, adapter, ranking_path)
    return _write_all(args.out, reader, args.src, res_specs, res_data, specs, by_name,
                      adapter, ranking_path, args.weights_id, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())