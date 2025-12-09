import pandas as pd
import pytest

def test_to_numeric_valid_input():
    s = pd.Series(['1.0', '2', -3])
    assert pd.to_numeric(s) == 1.0
    assert pd.to_numeric(s) == 2.0
    assert pd.to_numeric(s) == -3.0

def test_to_numeric_downcast_float():
    s = pd.Series(['1.0', '2', -3])
    assert pd.to_numeric(s, downcast='float') == 1.0
    assert pd.to_numeric(s, downcast='float') == 2.0
    assert pd.to_numeric(s, downcast='float') == -3.0

def test_to_numeric_downcast_signed():
    s = pd.Series(['1.0', '2', -3])
    assert pd.to_numeric(s, downcast='signed') == 1
    assert pd.to_numeric(s, downcast='signed') == 2
    assert pd.to_numeric(s, downcast='signed') == -3

def test_to_numeric_coerce_invalid():
    s = pd.Series(['1.0', 'abc', '2', -3])
    assert pd.to_numeric(s) == [1.0, pd.NA, 2.0, -3.0]

def test_to_numeric_ignore_invalid():
    s = pd.Series(['1.0', 'abc', '2', -3])
    assert pd.to_numeric(s, errors='ignore') == ['1.0', 'abc', '2', -3]

def test_to_numeric_empty_input():
    assert pd.to_numeric([]) == 0.0
    assert pd.to_numeric(pd.Series()) == 0.0

def test_to_numeric_mixed_dtypes():
    s = pd.Series(['1.0', '2', -3, 'abc'])
    assert pd.to_numeric(s) == [1.0, 2.0, -3.0, pd.NA]

def test_to_numeric_with_string_na():
    s = pd.Series(['1.0', '2', -3, 'NA'])
    assert pd.to_numeric(s) == [1.0, 2.0, -3.0, pd.NA]

def test_to_numeric_with_string_nan():
    s = pd.Series(['1.0', '2', -3, 'NaN'])
    assert pd.to_numeric(s) == [1.0, 2.0, -3.0, pd.NA]

def test_to_numeric_with_string_empty():
    s = pd.Series(['1.0', '2', -3, ''])
    assert pd.to_numeric(s) == [1.0, 2.0, -3.0, pd.NA]

def test_to_numeric_with_string_whitespace():
    s = pd.Series(['1.0', '2', -3, '   '])
    assert pd.to_numeric(s) == [1.0, 2.0, -3.0, pd.NA]

def test_to_numeric_with_string_leading_trailing_spaces():
    s = pd.Series([' 1.0 ', '2 ', '-3'])
    assert pd.to_numeric(s) == [1.0, 2.0, -3.0]

def test_to_numeric_with_string_non_numeric():
    s = pd.Series(['1.0', 'abc', '2', -3])
    assert pd.to_numeric(s, errors='coerce') == [1.0, pd.NA, 2.0, -3.0]

def test_to_numeric_with_string_non_numeric_ignore():
    s = pd.Series(['1.0', 'abc', '2', -3])
    assert pd.to_numeric(s, errors='ignore') == ['1.0', 'abc', '2', -3]

def test_to_numeric_with_large_numbers():
    s = pd.Series([10000000000.0, 20000000000.0])
    assert pd.to_numeric(s) == [1e+09, 2e+09]

def test_to_numeric_with_negative_large_numbers():
    s = pd.Series([-10000000000.0, -20000000000.0])
    assert pd.to_numeric(s) == [-1e+09, -2e+09]

def test_to_numeric_with_zero():
    s = pd.Series([0.0, 0, 0.0])
    assert pd.to_numeric(s) == [0.0, 0.0, 0.0]

def test_to_numeric_with_decimal():
    s = pd.Series([1.23456789, 2.34567890])
    assert pd.to_numeric(s) == [1.23456789, 2.34567890]

def test_to_numeric_with_boolean():
    s = pd.Series([True, False])
    assert pd.to_numeric(s) == [1, 0]

def test_to_numeric_with_datetime():
    s = pd.Series([pd.to_datetime('2023-10-26'), pd.to_datetime('2023-10-27')])
    assert pd.to_numeric(s) == [0.0, 1.0]