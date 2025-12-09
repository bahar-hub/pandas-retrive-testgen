import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

def test_eval_basic_arithmetic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = pd.eval("df['a'] + df['b']")
    expected = pd.Series([5, 7, 9], name="a + b")
    assert_series_equal(result, expected)

def test_eval_with_target():
    df = pd.DataFrame({"a": [1, 2, 3]})
    pd.eval("df['b'] = df['a'] * 2", target=df)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    assert_frame_equal(df, expected)

def test_eval_with_inplace():
    df = pd.DataFrame({"a": [1, 2, 3]})
    pd.eval("df['b'] = df['a'] * 2", target=df, inplace=True)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    assert_frame_equal(df, expected)

def test_eval_with_multiline():
    df = pd.DataFrame({"a": [1, 2, 3]})
    expr = "df['b'] = df['a'] * 2\n"
    expr += "df['c'] = df['a'] + 1"
    pd.eval(expr, target=df)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": [2, 3, 4]})
    assert_frame_equal(df, expected)

def test_eval_with_parser_python():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = pd.eval("df['a'] + df['b']", parser="python")
    expected = pd.Series([5, 7, 9], name="a + b")
    assert_series_equal(result, expected)

def test_eval_with_engine_python():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = pd.eval("df['a'] + df['b']", engine="python")
    expected = pd.Series([5, 7, 9], name="a + b")
    assert_series_equal(result, expected)

def test_eval_with_local_dict():
    df = pd.DataFrame({"a": [1, 2, 3]})
    local_dict = {"x": 10}
    result = pd.eval("x + df['a']", local_dict=local_dict)
    expected = pd.Series([11, 12, 13], name="x + a")
    assert_series_equal(result, expected)

def test_eval_with_global_dict():
    df = pd.DataFrame({"a": [1, 2, 3]})
    global_dict = {"x": 10}
    result = pd.eval("x + df['a']", global_dict=global_dict)
    expected = pd.Series([11, 12, 13], name="x + a")
    assert_series_equal(result, expected)

def test_eval_with_resolvers():
    df = pd.DataFrame({"a": [1, 2, 3]})
    resolvers = [{"x": 10}]
    result = pd.eval("x + df['a']", resolvers=resolvers)
    expected = pd.Series([11, 12, 13], name="x + a")
    assert_series_equal(result, expected)

def test_eval_with_level():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = pd.eval("df['a'] * 2", level=1)
    expected = pd.Series([2, 4, 6], name="a * 2")
    assert_series_equal(result, expected)

def test_eval_empty_dataframe():
    df = pd.DataFrame()
    result = pd.eval("df['a'] + 1")
    expected = pd.Series([], dtype="int64")
    assert_series_equal(result, expected)

def test_eval_empty_series():
    s = pd.Series([])
    result = pd.eval("s + 1")
    expected = pd.Series([], dtype="int64")
    assert_series_equal(result, expected)

def test_eval_with_string_operations():
    df = pd.DataFrame({"a": ["a", "b", "c"]})
    result = pd.eval("df['a'] + '_suffix'")
    expected = pd.Series(["a_suffix", "b_suffix", "c_suffix"], name="a + _suffix")
    assert_series_equal(result, expected)

def test_eval_with_boolean_operations():
    df = pd.DataFrame({"a": [True, False, True]})
    result = pd.eval("~df['a']")
    expected = pd.Series([False, True, False], name="~a")
    assert_series_equal(result, expected)

def test_eval_with_mixed_dtypes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = pd.eval("df['a'] * 2 + df['b']")
    expected = pd.Series(["1x", "2y", "3z"], name="a * 2 + b")
    assert_series_equal(result, expected)

def test_eval_with_nan_values():
    df = pd.DataFrame({"a": [1, 2, np.nan]})
    result = pd.eval("df['a'] * 2")
    expected = pd.Series([2, 4, np.nan], name="a * 2")
    assert_series_equal(result, expected)

def test_eval_with_invalid_expression():
    with pytest.raises(ValueError):
        pd.eval("x = 1 + 1")

def test_eval_with_invalid_target():
    with pytest.raises(ValueError):
        pd.eval("df['b'] = df['a'] * 2", target="invalid")

def test_eval_with_invalid_inplace():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        pd.eval("df['a'] * 2", inplace=True)

def test_eval_with_invalid_parser():
    with pytest.raises(ValueError):
        pd.eval("df['a'] + 1", parser="invalid")

def test_eval_with_invalid_engine():
    with pytest.raises(ValueError):
        pd.eval("df['a'] + 1", engine="invalid")