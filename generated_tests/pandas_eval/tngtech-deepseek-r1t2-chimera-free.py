import pytest
import pandas as pd
from pandas import Timedelta, DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

def test_basic_arithmetic_evaluation():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    result = pd.eval("a + b", target=df)
    expected = df["a"] + df["b"]
    assert_series_equal(result, expected)

def test_assignment_with_target_copy():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    original = df.copy()
    result = pd.eval("c = a + b", target=df)
    expected = original.assign(c=[4, 6])
    assert_frame_equal(result, expected)
    assert "c" not in df.columns

def test_inplace_assignment():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    expected = df.assign(c=[4, 6])
    pd.eval("c = a + b", target=df, inplace=True)
    assert_frame_equal(df, expected)

def test_multi_line_assignment():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    expr = """
    c = a + b
    d = c * 2
    """
    result = pd.eval(expr, target=df)
    expected = df.assign(c=[4, 6], d=[8, 12])
    assert_frame_equal(result, expected)

def test_boolean_expression_pandas_parser():
    df = DataFrame({"a": [True, False], "b": [False, True]})
    result = pd.eval("a and b", parser="pandas", target=df)
    expected = df["a"] & df["b"]
    assert_series_equal(result, expected)

def test_python_parser_bitwise_ops():
    df = DataFrame({"a": [True, False], "b": [False, True]})
    result = pd.eval("a & b", parser="python", target=df)
    expected = df["a"] & df["b"]
    assert_series_equal(result, expected)

def test_engine_fallback_to_python(capsys):
    df = DataFrame({"a": pd.Categorical([1, 2])})
    pd.eval("a + 1", target=df)
    captured = capsys.readouterr()
    assert "Engine has switched to 'python'" in captured.out

def test_local_dict_override():
    df = DataFrame({"a": [1, 2]})
    result = pd.eval("a + b", local_dict={"b": 3}, target=df)
    expected = df["a"] + 3
    assert_series_equal(result, expected)

def test_global_dict_usage():
    df = DataFrame({"a": [1, 2]})
    result = pd.eval("a + b", global_dict={"b": 3}, target=df)
    expected = df["a"] + 3
    assert_series_equal(result, expected)

def test_resolvers():
    class CustomResolver:
        def __getitem__(self, key):
            return Series([10, 20]) if key == "b" else None
    df = DataFrame({"a": [1, 2]})
    result = pd.eval("a + b", resolvers=[CustomResolver()], target=df)
    expected = df["a"] + Series([10, 20])
    assert_series_equal(result, expected)

def test_datetime_operations():
    df = DataFrame({"dates": pd.date_range("2020-01-01", periods=2)})
    result = pd.eval("dates + Timedelta('1D')", target=df)
    expected = df["dates"] + Timedelta(days=1)
    assert_series_equal(result, expected)

def test_empty_dataframe_valid_expression():
    df = DataFrame()
    result = pd.eval("1 + 2", target=df)
    assert result == 3

def test_multiline_without_target_raises():
    expr = """
    a = 1 + 2
    b = a + 3
    """
    with pytest.raises(ValueError, match="multi-line expressions"):
        pd.eval(expr)

def test_inplace_without_assignment_raises():
    df = DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Cannot operate inplace"):
        pd.eval("a + 2", target=df, inplace=True)

def test_invalid_assignment_target_raises():
    target = [1, 2, 3]
    with pytest.raises(ValueError, match="Cannot assign expression output"):
        pd.eval("x = 1", target=target)

def test_mixed_dtype_operations():
    df = DataFrame({"int": [1, 2], "str": ["a", "b"], "bool": [True, False]})
    result = pd.eval("int + 5", target=df)
    expected = df["int"] + 5
    assert_series_equal(result, expected)

def test_no_target_return_value():
    result = pd.eval("1 + 2 * 3")
    assert result == 7

def test_multi_index_access():
    df = DataFrame(
        [[1, 2], [3, 4]],
        columns=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")])
    )
    result = pd.eval("a.x + b.y", target=df)
    expected = df[("a", "x")] + df[("b", "y")]
    assert_series_equal(result, expected)

def test_empty_expression_raises():
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        pd.eval("")

def test_invalid_parser_raises():
    df = DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Invalid parser"):
        pd.eval("a + 1", parser="invalid", target=df)

def test_invalid_engine_raises():
    df = DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Invalid engine"):
        pd.eval("a + 1", engine="invalid", target=df)