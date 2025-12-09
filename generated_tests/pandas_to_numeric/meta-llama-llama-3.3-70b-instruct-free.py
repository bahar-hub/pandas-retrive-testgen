import pytest
import pandas as pd
import numpy as np

def test_to_numeric_series():
    s = pd.Series(['1.0', '2', '-3'])
    result = pd.to_numeric(s)
    expected = pd.Series([1.0, 2.0, -3.0], dtype='float64')
    pd.testing.assert_series_equal(result, expected)

def test_to_numeric_series_downcast_float():
    s = pd.Series(['1.0', '2', '-3'])
    result = pd.to_numeric(s, downcast='float')
    expected = pd.Series([1.0, 2.0, -3.0], dtype='float32')
    pd.testing.assert_series_equal(result, expected)

def test_to_numeric_series_downcast_signed():
    s = pd.Series(['1', '2', '-3'])
    result = pd.to_numeric(s, downcast='signed')
    expected = pd.Series([1, 2, -3], dtype='int8')
    pd.testing.assert_series_equal(result, expected)

def test_to_numeric_series_errors_coerce():
    s = pd.Series(['apple', '1.0', '2', '-3'])
    result = pd.to_numeric(s, errors='coerce')
    expected = pd.Series([np.nan, 1.0, 2.0, -3.0], dtype='float64')
    pd.testing.assert_series_equal(result, expected)

def test_to_numeric_series_errors_raise():
    s = pd.Series(['apple', '1.0', '2', '-3'])
    with pytest.raises(ValueError):
        pd.to_numeric(s, errors='raise')

def test_to_numeric_series_empty():
    s = pd.Series([])
    result = pd.to_numeric(s)
    expected = pd.Series([], dtype='float64')
    pd.testing.assert_series_equal(result, expected)

def test_to_numeric_series_non_numeric():
    s = pd.Series(['a', 'b', 'c'])
    result = pd.to_numeric(s, errors='coerce')
    expected = pd.Series([np.nan, np.nan, np.nan], dtype='float64')
    pd.testing.assert_series_equal(result, expected)

def test_to_numeric_list():
    lst = ['1.0', '2', '-3']
    result = pd.to_numeric(lst)
    expected = np.array([1.0, 2.0, -3.0], dtype='float64')
    np.testing.assert_array_equal(result, expected)

def test_to_numeric_list_downcast_float():
    lst = ['1.0', '2', '-3']
    result = pd.to_numeric(lst, downcast='float')
    expected = np.array([1.0, 2.0, -3.0], dtype='float32')
    np.testing.assert_array_equal(result, expected)

def test_to_numeric_list_downcast_signed():
    lst = ['1', '2', '-3']
    result = pd.to_numeric(lst, downcast='signed')
    expected = np.array([1, 2, -3], dtype='int8')
    np.testing.assert_array_equal(result, expected)

def test_to_numeric_list_errors_coerce():
    lst = ['apple', '1.0', '2', '-3']
    result = pd.to_numeric(lst, errors='coerce')
    expected = np.array([np.nan, 1.0, 2.0, -3.0], dtype='float64')
    np.testing.assert_array_equal(result, expected)

def test_to_numeric_list_errors_raise():
    lst = ['apple', '1.0', '2', '-3']
    with pytest.raises(ValueError):
        pd.to_numeric(lst, errors='raise')

def test_to_numeric_list_empty():
    lst = []
    result = pd.to_numeric(lst)
    expected = np.array([], dtype='float64')
    np.testing.assert_array_equal(result, expected)

def test_to_numeric_list_non_numeric():
    lst = ['a', 'b', 'c']
    result = pd.to_numeric(lst, errors='coerce')
    expected = np.array([np.nan, np.nan, np.nan], dtype='float64')
    np.testing.assert_array_equal(result, expected)

def test_to_numeric_scalar():
    scalar = '1.0'
    result = pd.to_numeric(scalar)
    expected = 1.0
    assert result == expected

def test_to_numeric_scalar_downcast_float():
    scalar = '1.0'
    result = pd.to_numeric(scalar, downcast='float')
    expected = 1.0
    assert result == expected

def test_to_numeric_scalar_downcast_signed():
    scalar = '1'
    result = pd.to_numeric(scalar, downcast='signed')
    expected = 1
    assert result == expected

def test_to_numeric_scalar_errors_coerce():
    scalar = 'apple'
    result = pd.to_numeric(scalar, errors='coerce')
    expected = np.nan
    assert np.isnan(result)

def test_to_numeric_scalar_errors_raise():
    scalar = 'apple'
    with pytest.raises(ValueError):
        pd.to_numeric(scalar, errors='raise')