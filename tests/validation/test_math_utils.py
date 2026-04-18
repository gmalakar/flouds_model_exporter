import numpy as np

from model_exporter.validation.math_utils import compare_arrays, mean_pooling, rowwise_cosine



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
