#!/usr/bin/env python3
"""Estimate KV-cache tensor sizes from the DeepSeek-V2-Lite dimensions.

This is a storage-arithmetic demo, not a model implementation or benchmark.
Only Python's standard library is required.
"""

import argparse


LAYERS = 27
HEADS = 16
HEAD_DIM = 128
KV_COMPRESSION_DIM = 512
ROPE_DIM = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768,
                        help="number of cached context tokens (default: 32768)")
    parser.add_argument("--dtype-bytes", type=int, default=2,
                        help="bytes per cache element, e.g. 2 for FP16/BF16")
    args = parser.parse_args()
    if args.tokens <= 0 or args.dtype_bytes <= 0:
        parser.error("--tokens and --dtype-bytes must be positive integers")

    elements_per_token = {
        "MHA (16 KV heads)": 2 * HEADS * HEAD_DIM * LAYERS,
        "GQA (8 KV groups)": 2 * 8 * HEAD_DIM * LAYERS,
        "MQA (1 KV group)": 2 * HEAD_DIM * LAYERS,
        "MLA (512 latent + 64 RoPE)":
            (KV_COMPRESSION_DIM + ROPE_DIM) * LAYERS,
    }

    print("DeepSeek-V2-Lite cache dimensions: "
          f"layers={LAYERS}, heads={HEADS}, head_dim={HEAD_DIM}, "
          f"d_c={KV_COMPRESSION_DIM}, d_h^R={ROPE_DIM}")
    print(f"Context tokens: {args.tokens}; bytes per element: {args.dtype_bytes}")
    print(f"{'Attention':32} {'elements/token':>16} {'KiB/token':>12} {'GiB/context':>13}")
    for name, elements in elements_per_token.items():
        bytes_per_token = elements * args.dtype_bytes
        context_bytes = bytes_per_token * args.tokens
        print(f"{name:32} {elements:16,d} "
              f"{bytes_per_token / 1024:12.3f} "
              f"{context_bytes / (1024 ** 3):13.3f}")

    mla_ratio = elements_per_token["MLA (512 latent + 64 RoPE)"] / elements_per_token[
        "MHA (16 KV heads)"
    ]
    print(f"\nMLA/MHA cache-element ratio: {mla_ratio:.4%}")
    print("Scope: cache tensors only; excludes weights, allocator/page overhead, "
          "temporary activations, batching effects, and quantization metadata.")


if __name__ == "__main__":
    main()

