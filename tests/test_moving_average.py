from financial_python.simple_strategy import moving_average
import numpy as np
import pytest

def test_basic_moving_average():
    vals = [1, 2, 3, 4, 5]
    out = moving_average(vals, window=3)
    expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0], dtype=float)
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_window_one_returns_copy_and_type():
    vals = (10, 20, 30)
    out = moving_average(vals, window=1)
    expected = np.array([10.0, 20.0, 30.0], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_window_larger_than_length_all_nans():
    vals = [1, 2]
    out = moving_average(vals, window=5)
    expected = np.array([np.nan, np.nan], dtype=float)
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_empty_input_returns_empty_array():
    out = moving_average([], window=3)
    expected = np.array([], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_negative_or_zero_window_raises():
    with pytest.raises(ValueError):
        moving_average([1, 2, 3], window=0)
    with pytest.raises(ValueError):
        moving_average([1, 2, 3], window=-2)


def test_numpy_input_and_precision():
    vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    out = moving_average(vals, window=2)
    expected = np.array([np.nan, 0.15, 0.25, 0.35], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0, equal_nan=True)