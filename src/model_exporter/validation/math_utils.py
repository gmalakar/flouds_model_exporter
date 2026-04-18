# =============================================================================
# File: validate_utils.py
# Date: 2026-01-09
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

"""Utility helpers for ONNX validation: pooling, normalization and comparisons.

This module extracts small pure-Python/numpy helpers to keep the main
validator script focused on orchestration.
"""

from __future__ import annotations

import numpy as np


def mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Compute mean-pooled sentence embeddings weighted by *attention_mask*.

    Averages token embeddings over the sequence dimension, ignoring padding
    tokens indicated by zero mask values.

    Args:
        last_hidden_state: Token embeddings of shape ``(batch, seq_len, hidden)``.
        attention_mask: Binary mask of shape ``(batch, seq_len)`` where ``1``
            marks real tokens and ``0`` marks padding.

    Returns:
        Sentence embeddings of shape ``(batch, hidden)``.
    """
    mask = attention_mask.astype(np.float32)
    mask = mask[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    denom = mask.sum(axis=1)
    denom = np.maximum(denom, 1e-9)
    return summed / denom


def compare_arrays(ref: np.ndarray, onnx_arr: np.ndarray) -> dict:
    """Compare two numpy arrays and return summary statistics.

    Returns a dict describing shape mismatch or element-wise differences.

    Args:
        ref: Reference array (typically from PyTorch).
        onnx_arr: Array to compare against the reference (from ONNX Runtime).

    Returns:
        A ``dict`` with fields:

        - ``shape_mismatch`` (``bool``): ``True`` when shapes differ.
        - ``ref_shape`` / ``onnx_shape``: Present only when shapes differ.
        - ``max_abs_diff``, ``mean_abs_diff``, ``l2``: Present when shapes match.
    """
    if ref.shape != onnx_arr.shape:
        return {
            "shape_mismatch": True,
            "ref_shape": ref.shape,
            "onnx_shape": onnx_arr.shape,
        }
    diff = np.abs(ref - onnx_arr)
    return {
        "shape_mismatch": False,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "l2": float(np.linalg.norm(ref - onnx_arr)),
    }


def l2_normalize(arr: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize *arr* along the given *axis*.

    Args:
        arr: Input array to normalize.
        axis: Axis along which to compute norms (default: ``-1``, last axis).
        eps: Small value to avoid division by zero (default: ``1e-12``).

    Returns:
        Normalized array with the same shape as *arr*. Returns *arr* unchanged
        if normalization fails.
    """
    try:
        norm = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)
        denom = np.maximum(norm, eps)
        return arr / denom
    except Exception:
        return arr


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity row-wise between two 2D arrays.

    Returns an array of shape (n_rows,) with cosine for each row.
    """
    ra = a.reshape((a.shape[0], -1))
    oa = b.reshape((b.shape[0], -1))
    rnorm = np.linalg.norm(ra, axis=1)
    onorm = np.linalg.norm(oa, axis=1)
    denom = rnorm * onorm
    denom = np.where(denom == 0, 1e-12, denom)
    return np.sum(ra * oa, axis=1) / denom
