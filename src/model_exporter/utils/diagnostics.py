# =============================================================================
# File: diagnostics.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""Diagnostics and dump collection for ONNX validation runs.

This module centralizes the logic that writes `validation_dumps` files and
creates a human-readable `validation_summary.txt` and optional post-opt
preservation artifacts.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from model_exporter.validation.math_utils import mean_pooling, rowwise_cosine


def collect_diagnostics(
    model_dir: str,
    results: Dict[str, Any],
    align_map: Dict[str, Any],
    tok_out: Dict[str, Any],
    ref_token_embeddings: np.ndarray,
    attention_mask: Any,
    normalize_flag: bool,
) -> str:
    """Write diagnostic arrays and a human-readable summary into
    `model_dir/validation_dumps`.

    Returns the dump directory path.
    """
    dump_dir = os.path.join(model_dir, "validation_dumps")
    os.makedirs(dump_dir, exist_ok=True)

    # Persist the texts/metadata if present in tok_out (caller may also
    # have written validation_texts.json; this is safe to overwrite)
    # (caller should write texts separately when available)

    # Find worst key for summary
    worst_key = None
    worst_val = -1.0
    for k, v in results.items():
        if isinstance(v, dict) and not v.get("shape_mismatch", False):
            val = float(v.get("max_abs_diff", 0.0))
            if val > worst_val:
                worst_val = val
                worst_key = k

    # Save aligned arrays for the worst key when available
    try:
        if worst_key and worst_key in align_map:
            ref_arr, onnx_arr = align_map[worst_key]
            ref_path = os.path.join(dump_dir, f"{worst_key}_ref.npy")
            onnx_path = os.path.join(dump_dir, f"{worst_key}_onnx.npy")
            np.save(ref_path, ref_arr)
            np.save(onnx_path, onnx_arr)
            # per-sample cosine
            try:
                cos = rowwise_cosine(ref_arr, onnx_arr)
                cos_path = os.path.join(dump_dir, f"{worst_key}_cosine.npy")
                np.save(cos_path, cos)
            except Exception:
                pass
    except Exception:
        pass

    # Persist token-level inputs if available
    try:
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key in tok_out:
                val = tok_out[key]
                try:
                    arr = val.cpu().numpy() if hasattr(val, "cpu") else (val.numpy() if hasattr(val, "numpy") else np.array(val))
                except Exception:
                    try:
                        arr = np.array(val)
                    except Exception:
                        arr = None
                if isinstance(arr, np.ndarray):
                    try:
                        np.save(os.path.join(dump_dir, f"{key}.npy"), arr)
                    except Exception:
                        pass
    except Exception:
        pass

    # Save token-level and pooled reference embeddings
    try:
        token_ref_path = os.path.join(dump_dir, "token_embeddings_ref.npy")
        np.save(token_ref_path, ref_token_embeddings)
    except Exception:
        pass

    # Try to locate token-level ONNX array in align_map
    token_onnx = None
    try:
        for _k, v in align_map.items():
            try:
                _ref_v, _onnx_v = v
                if isinstance(_onnx_v, np.ndarray) and _onnx_v.shape == ref_token_embeddings.shape:
                    token_onnx = _onnx_v
                    break
            except Exception:
                continue
    except Exception:
        token_onnx = None

    if token_onnx is not None:
        try:
            np.save(os.path.join(dump_dir, "token_embeddings_onnx.npy"), token_onnx)
        except Exception:
            pass

    # Mean-pooled token embeddings using attention_mask
    try:
        if isinstance(attention_mask, np.ndarray):
            am = attention_mask
        else:
            try:
                am = attention_mask.cpu().numpy() if hasattr(attention_mask, "cpu") else np.array(attention_mask)
            except Exception:
                am = np.ones(ref_token_embeddings.shape[:2], dtype=np.int32)

        pooled_ref = mean_pooling(ref_token_embeddings, am)
        try:
            np.save(os.path.join(dump_dir, "token_embeddings_mean_ref.npy"), pooled_ref)
        except Exception:
            pass

        if token_onnx is not None:
            pooled_onnx = mean_pooling(token_onnx, am)
            try:
                np.save(
                    os.path.join(dump_dir, "token_embeddings_mean_onnx.npy"),
                    pooled_onnx,
                )
            except Exception:
                pass
            try:
                cos_pooled = rowwise_cosine(pooled_ref, pooled_onnx)
                np.save(
                    os.path.join(dump_dir, "token_embeddings_mean_cosine.npy"),
                    cos_pooled,
                )
            except Exception:
                pass
    except Exception:
        pass

    # Write validation_summary.txt
    try:
        summary_lines = []
        summary_lines.append(f"Validation summary for model_dir={model_dir}")
        summary_lines.append(f"Worst key: {worst_key}")
        if worst_key:
            v = results.get(worst_key)
            if isinstance(v, dict) and not v.get("shape_mismatch", False):
                summary_lines.append(f"  max_abs_diff: {v.get('max_abs_diff')}")
                summary_lines.append(f"  mean_abs_diff: {v.get('mean_abs_diff')}")
                summary_lines.append(f"  l2: {v.get('l2')}")

        try:
            ppath = os.path.join(dump_dir, "token_embeddings_mean_cosine.npy")
            if os.path.exists(ppath):
                pc = np.load(ppath)
                summary_lines.append(f"Pooled token cosine mean/min/max: {float(pc.mean()):.6f}/{float(pc.min()):.6f}/{float(pc.max()):.6f}")
        except Exception:
            pass

        try:
            summary_path = os.path.join(dump_dir, "validation_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as fh:
                for ln in summary_lines:
                    fh.write(ln + "\n")
        except Exception:
            pass
    except Exception:
        pass

    return dump_dir
