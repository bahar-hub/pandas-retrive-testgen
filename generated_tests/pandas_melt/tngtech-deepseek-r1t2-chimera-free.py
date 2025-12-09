import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

def test_basic_melt():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": ["x", "y"]})
    result = pd.melt(df, id_vars=["C"], value_vars=["A", "B"])
    expected = pd.DataFrame({
        "C": ["x", "y", "x", "y"],
        "variable": ["A", "A", "B", "B"],
        "value": [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_multiindex_columns():
    df = pd.DataFrame({("A", "X"): [1, 2], ("B", "Y"): [3, 4]})
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    result = pd.melt(df)
    expected = pd.DataFrame({
        "variable_0": ["A", "A", "B", "B"],
        "variable_1": ["X", "X", "Y", "Y"],
        "value": [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_col_level_parameter():
    df = pd.DataFrame({("A", "X"): [1], ("B", "Y"): [2]})
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["l1", "l2"])
    result = pd.melt(df, col_level="l1")
    expected = pd.DataFrame({
        "l1": ["A", "B"],
        "value": [1, 2]
    })
    assert_frame_equal(result, expected)

def test_value_name_conflict():
    df = pd.DataFrame({"A": [1], "value": [2]})
    with pytest.raises(ValueError, match="value_name \(value\) cannot match"):
        pd.melt(df, value_name="value")

def test_missing_columns():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(KeyError, match="not present in the DataFrame"):
        pd.melt(df, id_vars=["B"])

def test_ignore_index_false():
    df = pd.DataFrame({"A": [1, 2]}, index=["x", "y"])
    result = pd.melt(df, ignore_index=False)
    expected_index = pd.Index(["x", "y", "x", "y"])
    assert result.index.equals(expected_index)

def test_empty_dataframe():
    df = pd.DataFrame(columns=["A", "B"])
    result = pd.melt(df)
    assert result.empty
    assert result.columns.tolist() == ["variable", "value"]

def test_mixed_dtypes():
    df = pd.DataFrame({
        "A": [1, 2],
        "B": ["x", "y"],
        "C": pd.Categorical(["a", "b"])
    })
    result = pd.melt(df, id_vars=["C"])
    expected = pd.DataFrame({
        "C": pd.Categorical(["a", "b", "a", "b"]),
        "variable": ["A", "A", "B", "B"],
        "value": [1, 2, "x", "y"]
    })
    assert_frame_equal(result, expected)

def test_non_unique_index():
    df = pd.DataFrame({"A": [1, 2]}, index=[0, 0])
    result = pd.melt(df)
    expected = pd.DataFrame({
        "variable": ["A", "A"],
        "value": [1, 2]
    }, index=[0, 0])
    assert_frame_equal(result, expected)

def test_datetime_dtype():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2),
        "val": [10, 20]
    })
    result = pd.melt(df, id_vars=["date"])
    expected = pd.DataFrame({
        "date": df["date"].repeat(2).reset_index(drop=True),
        "variable": ["val", "val"],
        "value": [10, 20]
    })
    assert_frame_equal(result, expected)

def test_var_name_as_scalar():
    df = pd.DataFrame({"A": [1], "B": [2]})
    result = pd.melt(df, var_name="var")
    expected = pd.DataFrame({
        "var": ["A", "B"],
        "value": [1, 2]
    })
    assert_frame_equal(result, expected)

def test_var_name_list_like_raises():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError, match="var_name must be a scalar"):
        pd.melt(df, var_name=["var1", "var2"])

def test_no_value_vars():
    df = pd.DataFrame({"A": [1], "B": [2]})
    result = pd.melt(df, value_vars=[])
    assert result.columns.tolist() == ["variable", "value"]
    assert len(result) == 0

def test_id_vars_overlap_value_vars():
    df = pd.DataFrame({"A": [1], "B": [2]})
    result = pd.melt(df, id_vars=["A"], value_vars=["A", "B"])
    expected = pd.DataFrame({
        "A": [1],
        "variable": ["B"],
        "value": [2]
    })
    assert_frame_equal(result, expected)

def test_extension_dtype():
    df = pd.DataFrame({
        "A": pd.array([1, None], dtype="Int64"),
        "B": [3, 4]
    })
    result = pd.melt(df, id_vars=["A"])
    expected = pd.DataFrame({
        "A": pd.array([1, None], dtype="Int64"),
        "variable": ["B", "B"],
        "value": [3, 4]
    })
    assert_frame_equal(result, expected)

def test_preserves_value_var_order():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = pd.melt(df, value_vars=["C", "B"])
    expected = pd.DataFrame({
        "variable": ["C", "B"],
        "value": [3, 2]
    })
    assert_frame_equal(result, expected)

def test_all_columns_melted():
    df = pd.DataFrame({"A": [1], "B": [2]})
    result = pd.melt(df)
    expected = pd.DataFrame({
        "variable": ["A", "B"],
        "value": [1, 2]
    })
    assert_frame_equal(result, expected)

def test_does_not_modify_original():
    df = pd.DataFrame({"A": [1], "B": [2]})
    original = df.copy()
    result = pd.melt(df)
    result.iloc[0, 1] = 99
    assert_frame_equal(df, original)

def test_boolean_dtype():
    df = pd.DataFrame({"A": [True, False], "B": [False, True]})
    result = pd.melt(df)
    expected = pd.DataFrame({
        "variable": ["A", "A", "B", "B"],
        "value": [True, False, False, True]
    })
    assert_frame_equal(result, expected)

def test_multiindex_rows():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df.index = pd.MultiIndex.from_tuples([("x", 1), ("y", 2)])
    result = pd.melt(df, ignore_index=False)
    expected_index = pd.MultiIndex.from_tuples([
        ("x", 1), ("y", 2), ("x", 1), ("y", 2)
    ])
    assert result.index.equals(expected_index)