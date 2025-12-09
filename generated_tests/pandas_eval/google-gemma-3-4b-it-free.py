import pytest
import pandas as pd

try:
    pd.eval
except:
    pd.eval = lambda expr, **kwargs: expr  # Mock pd.eval for testing

def test_empty_expression():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert pd.eval("") == ""

def test_simple_addition():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = pd.eval("df.a + df.b")
    assert result.equals(pd.Series([4, 6]))

def test_dataframe_assignment():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    pd.eval("df.a + df.b", target=df)
    assert df["a"].equals(pd.Series([4, 6]))

def test_series_assignment():
    s = pd.Series([1, 2, 3])
    pd.eval("s + 1", target=s)
    assert s.equals(pd.Series([2, 3, 4]))

def test_string_expression():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = pd.eval("df.a * 2")
    assert result.equals(pd.Series([2, 4]))

def test_invalid_expression_syntax():
    with pytest.raises(ValueError):
        pd.eval("df.a +")

def test_invalid_expression_multiline_no_assignment():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        pd.eval("df.a + df.b + 1")

def test_inplace_modification():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    pd.eval("df.a + df.b", inplace=True)
    assert df["a"].equals(pd.Series([4, 6]))

def test_invalid_inplace_no_assignment():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        pd.eval("df.a + df.b", inplace=True)

def test_complex_expression():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = pd.eval("df.a * df.b + df.c")
    assert result.equals(pd.Series([ 4, 12]))

def test_mixed_dtypes():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = pd.eval("df.a + df.b")
    assert result.equals(pd.Series([3, "xy"]))

def test_empty_dataframe():
    df = pd.DataFrame()
    assert pd.eval("") == ""

def test_empty_series():
    s = pd.Series([])
    assert pd.eval("") == ""

def test_assignment_with_no_target():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        pd.eval("df.a + df.b + 1")

def test_assignment_with_target_not_support_copy():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        pd.eval("df.a + df.b", target=1)