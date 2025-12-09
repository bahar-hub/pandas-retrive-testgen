import pytest
import pandas as pd
import numpy as np
from pandas import Series, Index
from pandas.testing import assert_series_equal

def test_scalar_input():
    # Test scalar inputs
    assert pd.to_numeric(5) == 5
    assert pd.to_numeric(3.14) == 3.14
    assert pd.to_numeric("42") == 42
    assert pd.to_numeric("3.14") == 3.14

def test_list_input():
    # Test list inputs
    result = pd.to_numeric([1, "2", 3.0])
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, expected)

def test_series_basic():
    # Test basic Series conversion
    s = Series(["1.0", "2", -3])
    result = pd.to_numeric(s)
    expected = Series([1.0, 2.0, -3.0])
    assert_series_equal(result, expected)

def test_errors_coerce():
    # Test error handling with coerce
    s = Series(["apple", "1.0", "2", -3])
    result = pd.to_numeric(s, errors="coerce")
    expected = Series([np.nan, 1.0, 2.0, -3.0])
    assert_series_equal(result, expected)

def test_errors_raise():
    # Test error handling with raise
    s = Series(["apple", "1.0"])
    with pytest.raises(ValueError):
        pd.to_numeric(s, errors="raise")

def test_downcast_float():
    # Test downcast to float
    s = Series([1.0, 2.0, 3.0])
    result = pd.to_numeric(s, downcast="float")
    assert result.dtype == "float32"

def test_downcast_signed():
    # Test downcast to signed integer
    s = Series([1, 2, -3])
    result = pd.to_numeric(s, downcast="signed")
    assert result.dtype == "int8"

def test_downcast_unsigned():
    # Test downcast to unsigned integer
    s = Series([1, 2, 3])
    result = pd.to_numeric(s, downcast="unsigned")
    assert result.dtype == "uint8"

def test_downcast_invalid():
    # Test invalid downcast parameter
    with pytest.raises(ValueError):
        pd.to_numeric([1, 2], downcast="invalid")

def test_empty_input():
    # Test empty input handling
    assert pd.to_numeric([]).size == 0
    s = Series([], dtype="float64")
    result = pd.to_numeric(s)
    assert_series_equal(result, s)

def test_large_numbers():
    # Test very large numbers that might convert to float
    large_num = 2**64 + 1
    result = pd.to_numeric([large_num])
    assert result[0] == float(large_num)

def test_mixed_dtypes():
    # Test mixed dtype input
    data = [True, False, 1, 0.5, "3.14"]
    result = pd.to_numeric(data, errors="coerce")
    expected = np.array([1.0, 0.0, 1.0, 0.5, 3.14])
    np.testing.assert_allclose(result, expected)

def test_index_conversion():
    # Test Index input
    idx = Index(["1", "2", "3"])
    result = pd.to_numeric(idx)
    expected = Index([1, 2, 3], dtype="int64")
    pd.testing.assert_index_equal(result, expected)

def test_non_1d_input():
    # Test non-1D input raises error
    with pytest.raises(TypeError):
        pd.to_numeric([[1, 2], [3, 4]])

def test_nullable_dtypes():
    # Test nullable integer dtype
    s = Series([1, 2, None], dtype="Int64")
    result = pd.to_numeric(s, downcast="integer")
    assert result.dtype == "Int8"

def test_boolean_conversion():
    # Test boolean conversion
    s = Series([True, False])
    result = pd.to_numeric(s)
    expected = Series([1, 0], dtype="int64")
    assert_series_equal(result, expected)

def test_datetime_conversion():
    # Test datetime conversion to numeric
    dates = pd.date_range("2020-01-01", periods=3)
    result = pd.to_numeric(dates)
    expected = dates.asi8
    np.testing.assert_array_equal(result, expected)

def test_dtype_backend_pyarrow():
    # Test pyarrow dtype backend
    pytest.importorskip("pyarrow")
    s = Series(["1", "2", "3"])
    result = pd.to_numeric(s, dtype_backend="pyarrow")
    assert str(result.dtype) == "int64[pyarrow]"

def test_deprecated_ignore():
    # Test deprecated errors='ignore' behavior
    s = Series(["a", "1"])
    with pytest.warns(FutureWarning):
        result = pd.to_numeric(s, errors="ignore")
    assert_series_equal(result, s)