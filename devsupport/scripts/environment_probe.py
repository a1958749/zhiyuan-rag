"""Prove CUDA kernels and actual Qwen embeddings work; availability alone is insufficient."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def probe(model_path: Path, output: Path) -> dict:
    """Run tensor and embedding inference and preserve reproducible safe evidence."""
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU verification cannot use CPU fallback")
    total = torch.ones(16, device="cuda").sum().item()
    assert total == 16, "real CUDA tensor operation failed"
    texts = ["JSONLoader accepts jq_schema for parsing JSON records.", "Vector retrieval uses normalized embeddings."]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        last = model(**batch).last_hidden_state[:, -1]
        embeddings = torch.nn.functional.normalize(last.float(), p=2, dim=1).cpu().numpy()
    torch.cuda.synchronize()
    norms = np.linalg.norm(embeddings, axis=1)
    assert embeddings.shape == (2, 1024), "unexpected embedding dimensions"
    assert np.isfinite(embeddings).all() and np.allclose(norms, 1, atol=1e-5), "invalid normalized vectors"
    result = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "cuda_tensor_sum": total, "embedding_shape": list(embeddings.shape), "norms": norms.tolist(),
        "finite": True, "inference_device": str(next(model.parameters()).device),
        "attention": "sdpa", "dtype": str(next(model.parameters()).dtype), "max_length": 2048,
        "load_and_encode_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "original_failure": "global Torch 2.6.0+cu124 lacked sm_120 kernels; tensor operation failed",
        "fix": "isolated Torch 2.11.0+cu128; global interpreter/packages unchanged",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Run and record the real model probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.model, args.output), indent=2))


if __name__ == "__main__":
    main()
