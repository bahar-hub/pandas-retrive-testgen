import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

def test_to_numeric_scalar_integer():
    result = pd.to_numeric(42)
    assert result == 42

def test_to_numeric_scalar_float():
    result = pd.to_numeric(3.14)
    assert result == 3.14

def test_to_numeric_scalar_string_integer():
    result = pd.to_numeric("42")
    assert result == 42

def test_to_numeric_scalar_string_float():
    result = pd.to_numeric("3.14")
    assert result == 3.14

def test_to_numeric_list_integers():
    result = pd.to_numeric([1, 2, 3])
    assert (result == [1, 2, 3]).all()

def test_to_numeric_list_floats():
    result = pd.to_numeric([1.1, 2.2, 3.3])
    assert (result == [1.1, 2.2, 3.3]).all()

def test_to_numeric_list_strings():
    result = pd.to_numeric(["1", "2", "3"])
    assert (result == [1, 2, 3]).all()

def test_to_numeric_series_integers():
    s = pd.Series([1, 2, 3])
    result = pd.to_numeric(s)
    assert_series_equal(result, s)

def test_to_numeric_series_floats():
    s = pd.Series([1.1, 2.2, 3.3])
    result = pd.to_numeric(s)
    assert_series_equal(result, s)

def test_to_numeric_series_strings():
    s = pd.Series(["1", "2", "3"])
    result = pd.to_numeric(s)
    assert_series_equal(result, pd.Series([1, 2, 3]))

def test_to_numeric_errors_raise():
    with pytest.raises(ValueError):
        pd.to_numeric(["1", "two", "3"], errors="raise")

def test_to_numeric_errors_coerce():
    result = pd.to_numeric(["1", "two", "3"], errors="coerce")
    assert (result == [1, np.nan, 3]).all()

def test_to_numeric_errors_ignore():
    with pytest.warns(FutureWarning):
        result = pd.to_numeric(["1", "two", "3"], errors="ignore")
    assert (result == ["1", "two", "3"]).all()

def test_to_numeric_downcast_integer():
    s = pd.Series([1, 2, 3])
    result = pd.to_numeric(s, downcast="integer")
    assert result.dtype == np.int8

def test_to_numeric_downcast_signed():
    s = pd.Series([1, 2, 3])
    result = pd.to_numeric(s, downcast="signed")
    assert result.dtype == np.int8

def test_to_numeric_downcast_unsigned():
    s = pd.Series([1, 2, 3])
    result = pd.to_numeric(s, downcast="unsigned")
    assert result.dtype == np.uint8

def test_to_numeric_downcast_float():
    s = pd.Series([1.1, 2.2, 3.3])
    result = pd.to_numeric(s, downcast="float")
    assert result.dtype == np.float32

def test_to_numeric_empty_series():
    s = pd.Series([])
    result = pd.to_numeric(s)
    assert result.dtype == np.float64

def test_to_numeric_empty_list():
    result = pd.to_numeric([])
    assert result.dtype == np.float64

def test_to_numeric_boolean():
    result = pd.to_numeric(True)
    assert result == 1

def test_to_numeric_boolean_series():
    s = pd.Series([True, False, True])
    result = pd.to_numeric(s)
    assert (result == [1, 0, 1]).all()

def test_to_numeric_mixed_types():
    s = pd.Series(["1", 2, 3.0, "4.5"])
    result = pd.to_numeric(s)
    assert (result == [1, 2, 3.0, 4.5]).all()

def test_to_numeric_invalid_downcast():
    with pytest.raises(ValueError):
        pd.to_numeric([1, 2, 3], downcast="invalid")

def test_to_numeric_invalid_errors():
    with pytest.raises(ValueError):
        pd.to_numeric([1, 2, 3], errors="invalid")

def test_to_numeric_large_numbers():
    s = pd.Series([1e100, 2e100, 3e100])
    result = pd.to_numeric(s)
    assert (result == [1e100, 2e100, 3e100]).all()

def test_to_numeric_small_numbers():
    s = pd.Series([1e-100, 2e-100, 3e-100])
    result = pd.to_numeric(s)
    assert (result == [1e-100, 2e-100, 3e-100]).all()

def test_to_numeric_nullable_integer():
    s = pd.Series([1, 2, 3], dtype="Int64")
    result = pd.to_numeric(s, downcast="integer")
    assert result.dtype == "Int8"

def test_to_numeric_nullable_float():
    s = pd.Series([1.0, 2.1, 3.0], dtype="Float64")
    result = pd.to_numeric(s, downcast="float")
    assert result.dtype == "Float32"