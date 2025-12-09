import pytest
import pandas as pd
import numpy as np

def test_eval_single_line_expression():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("df.age * 2")
    expected = pd.Series([20, 40])
    pd.testing.assert_series_equal(result, expected)

def test_eval_multi_line_expression():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df)
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_inplace_true():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, inplace=True)
    assert result is None
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(df, expected)

def test_eval_with_inplace_false():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, inplace=False)
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_parser_pandas():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, parser="pandas")
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_parser_python():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, parser="python")
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_engine_numexpr():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, engine="numexpr")
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_engine_python():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, engine="python")
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_local_dict():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    local_dict = {"df": df}
    result = pd.eval("double_age = df.age * 2", target=df, local_dict=local_dict)
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_global_dict():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    global_dict = {"df": df}
    result = pd.eval("double_age = df.age * 2", target=df, global_dict=global_dict)
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_resolvers():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    resolvers = [{"df": df}]
    result = pd.eval("double_age = df.age * 2", target=df, resolvers=resolvers)
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_level():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, level=1)
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_target():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df)
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(result, expected)

def test_eval_with_inplace_and_target():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    result = pd.eval("double_age = df.age * 2", target=df, inplace=True)
    assert result is None
    expected = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20], "double_age": [20, 40]})
    pd.testing.assert_frame_equal(df, expected)

def test_eval_with_invalid_parser():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(ValueError):
        pd.eval("double_age = df.age * 2", target=df, parser="invalid")

def test_eval_with_invalid_engine():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(ValueError):
        pd.eval("double_age = df.age * 2", target=df, engine="invalid")

def test_eval_with_invalid_target():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(ValueError):
        pd.eval("double_age = df.age * 2", target=None)

def test_eval_with_invalid_inplace():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(ValueError):
        pd.eval("df.age * 2", target=df, inplace=True)

def test_eval_with_empty_expression():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(ValueError):
        pd.eval("", target=df)

def test_eval_with_invalid_expression():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(Exception):
        pd.eval("invalid_expression", target=df)

def test_eval_with_multi_line_expression_without_assignment():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(ValueError):
        pd.eval("df.age * 2\ndf.age * 3", target=df)

def test_eval_with_multi_line_expression_with_incomplete_assignment():
    df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    with pytest.raises(ValueError):
        pd.eval("double_age = df.age * 2\ndf.age * 3", target=df)