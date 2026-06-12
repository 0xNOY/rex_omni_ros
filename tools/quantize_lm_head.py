#!/usr/bin/env python3
"""Quantize the lm_head of Rex-Omni-AWQ to AWQ 4-bit (group 128, zero-point).

The published AWQ checkpoint ties lm_head to the fp16 embedding table
(151936 x 2048 = 622 MB), which the decoder re-reads on every generated
token; on an RTX 3090 that is ~25% of the per-token weight traffic. This
script unties the head, round-to-nearest quantizes it into the standard
AutoAWQ GEMM layout (qweight / qzeros / scales) and writes a local model
directory with ``tie_word_embeddings=false`` and ``lm_head: true`` in the
quantization config, which vLLM's awq_marlin backend picks up natively.
The embedding table itself stays fp16: input lookups are row gathers and
need no GEMM.

Usage:
    pixi run python tools/quantize_lm_head.py \
        --model IDEA-Research/Rex-Omni-AWQ \
        --output models/Rex-Omni-AWQ-QLMHead
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

EMBED_KEY = "model.embed_tokens.weight"
# AutoAWQ GEMM nibble interleave: nibble i of a packed int32 holds output
# channel ``col * 8 + ORDER_MAP[i]``.
ORDER_MAP = [0, 2, 4, 6, 1, 3, 5, 7]


def awq_quantize(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit RTN per input-dim group (AutoAWQ convention).

    Returns ``(q, zeros, scales)`` with ``q``: [out, in] int32 in [0, 15],
    ``zeros``/``scales``: [in // group_size, out].
    """
    out_features, in_features = weight.shape
    grouped = weight.float().reshape(out_features, in_features // group_size, group_size)
    max_val = grouped.amax(dim=2)
    min_val = grouped.amin(dim=2)
    scales = (max_val - min_val).clamp(min=1e-5) / 15.0
    zeros = (-torch.round(min_val / scales)).clamp(0, 15)
    q = torch.round(grouped / scales.unsqueeze(2)) + zeros.unsqueeze(2)
    q = q.clamp(0, 15).reshape(out_features, in_features).to(torch.int32)
    return q, zeros.t().contiguous(), scales.t().contiguous()


def awq_pack(values: torch.Tensor) -> torch.Tensor:
    """Pack int values in [0, 15] along the last dim, 8 per int32."""
    rows, cols = values.shape
    grouped = values.reshape(rows, cols // 8, 8)[:, :, ORDER_MAP]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    return (grouped << shifts).sum(dim=2, dtype=torch.int32)


def dequantize(
    q: torch.Tensor, zeros: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    out_features, in_features = q.shape
    z = zeros.t().reshape(out_features, in_features // group_size, 1)
    s = scales.t().reshape(out_features, in_features // group_size, 1)
    grouped = q.reshape(out_features, in_features // group_size, group_size)
    return ((grouped.float() - z) * s).reshape(out_features, in_features)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("--model", default="IDEA-Research/Rex-Omni-AWQ")
    arg_parser.add_argument("--output", default="models/Rex-Omni-AWQ-QLMHead")
    arg_parser.add_argument("--group-size", type=int, default=128)
    args = arg_parser.parse_args()

    source = Path(snapshot_download(args.model))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text())
    quant_config = config["quantization_config"]
    if quant_config.get("lm_head"):
        raise SystemExit(f"{args.model} already has a quantized lm_head")

    state = load_file(source / "model.safetensors")
    weight = state[EMBED_KEY]
    print(f"lm_head source: {EMBED_KEY} {tuple(weight.shape)} {weight.dtype}")

    q, zeros, scales = awq_quantize(weight, args.group_size)
    error = (dequantize(q, zeros, scales, args.group_size) - weight.float()).abs()
    scale_ref = weight.float().abs().mean()
    print(
        f"RTN error: mean {error.mean():.2e}, max {error.max():.2e} "
        f"(weight mean abs {scale_ref:.2e})"
    )

    state["lm_head.qweight"] = awq_pack(q.t().contiguous())
    state["lm_head.qzeros"] = awq_pack(zeros.to(torch.int32))
    state["lm_head.scales"] = scales.to(weight.dtype).contiguous()

    config["tie_word_embeddings"] = False
    quant_config["lm_head"] = True
    quant_config["modules_to_not_convert"] = [
        name for name in quant_config["modules_to_not_convert"] if name != "lm_head"
    ]

    for path in source.iterdir():
        if path.is_file() and path.name not in {"model.safetensors", "config.json"}:
            shutil.copy(path, output / path.name)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(state, output / "model.safetensors", metadata={"format": "pt"})

    packed_mb = sum(
        state[k].numel() * state[k].element_size() for k in
        ("lm_head.qweight", "lm_head.qzeros", "lm_head.scales")
    ) / 2**20
    print(f"lm_head packed size: {packed_mb:.0f} MB (was {weight.numel() * 2 / 2**20:.0f} MB fp16)")
    print(f"written to {output}")


if __name__ == "__main__":
    main()
