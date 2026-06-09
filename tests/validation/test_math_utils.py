# =============================================================================
# File: test_math_utils.py
# Date: 2026-04-18
# Copyright (c) 2026 Goutam Malakar.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

import numpy as np

from model_exporter.validation.math_utils import (
    compare_arrays,
    comparisons_within_tolerance,
    mean_pooling,
    rowwise_cosine,
    summarize_comparison_results,
)


def test_mean_pooling_respects_attention_mask():
    hidden = np.array(
        [
            [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[1, 1, 0]], dtype=np.int64)

    pooled = mean_pooling(hidden, mask)

    np.testing.assert_allclose(pooled, np.array([[2.0, 2.0]], dtype=np.float32))


def test_compare_arrays_reports_shape_mismatch():
    ref = np.zeros((2, 3), dtype=np.float32)
    onnx_arr = np.zeros((2, 4), dtype=np.float32)

    result = compare_arrays(ref, onnx_arr)

    assert result == {
        "shape_mismatch": True,
        "ref_shape": (2, 3),
        "onnx_shape": (2, 4),
    }


def test_rowwise_cosine_returns_one_for_identical_rows():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    cosine = rowwise_cosine(arr, arr)

    np.testing.assert_allclose(cosine, np.ones(2, dtype=np.float32))


def test_summarize_comparison_results_requires_comparable_output():
    results = {
        "sentence_embedding": {
            "shape_mismatch": True,
            "ref_shape": (1, 3),
            "onnx_shape": (1, 4),
        }
    }

    max_diff, comparable_count, failures = summarize_comparison_results(results)

    assert max_diff == 0.0
    assert comparable_count == 0
    assert failures == ["sentence_embedding"]


def test_summarize_comparison_results_tracks_max_diff():
    results = {
        "token_embeddings": {
            "shape_mismatch": False,
            "max_abs_diff": 0.001,
        },
        "sentence_embedding": {
            "shape_mismatch": False,
            "max_abs_diff": 0.003,
        },
    }

    max_diff, comparable_count, failures = summarize_comparison_results(results)

    assert max_diff == 0.003
    assert comparable_count == 2
    assert failures == []


def test_comparisons_within_tolerance_uses_relative_tolerance():
    results = {
        "large_values": compare_arrays(
            np.array([1000.0], dtype=np.float32),
            np.array([1000.5], dtype=np.float32),
        )
    }

    assert comparisons_within_tolerance(results, atol=1e-4, rtol=1e-3) is True


def test_comparisons_within_tolerance_rejects_shape_mismatch():
    results = {
        "sentence_embedding": {
            "shape_mismatch": True,
            "ref_shape": (1, 3),
            "onnx_shape": (1, 4),
        }
    }

    assert comparisons_within_tolerance(results, atol=1.0, rtol=1.0) is False
