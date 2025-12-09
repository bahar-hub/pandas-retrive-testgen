import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

def test_melt_empty_dataframe():
    df = pd.DataFrame()
    result = pd.melt(df)
    assert_frame_equal(result, pd.DataFrame(columns=['variable', 'value']))

def test_melt_single_column():
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = pd.melt(df)
    expected = pd.DataFrame({'variable': ['A', 'A', 'A'], 'value': [1, 2, 3]})
    assert_frame_equal(result, expected)

def test_melt_multiple_columns():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_id_vars():
    df = pd.DataFrame({'id': [1, 2, 3], 'A': [4, 5, 6], 'B': [7, 8, 9]})
    result = pd.melt(df, id_vars='id')
    expected = pd.DataFrame({
        'id': [1, 1, 2, 2, 3, 3],
        'variable': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [4, 7, 5, 8, 6, 9]
    })
    assert_frame_equal(result, expected)

def test_melt_value_vars():
    df = pd.DataFrame({'id': [1, 2, 3], 'A': [4, 5, 6], 'B': [7, 8, 9]})
    result = pd.melt(df, value_vars=['A', 'B'])
    expected = pd.DataFrame({
        'id': [1, 1, 2, 2, 3, 3],
        'variable': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [4, 7, 5, 8, 6, 9]
    })
    assert_frame_equal(result, expected)

def test_melt_var_name():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.melt(df, var_name='column')
    expected = pd.DataFrame({
        'column': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_value_name():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.melt(df, value_name='val')
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'val': [1, 2, 3, 4, 5, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_col_level():
    df = pd.DataFrame({
        ('A', 'x'): [1, 2, 3],
        ('A', 'y'): [4, 5, 6],
        ('B', 'x'): [7, 8, 9],
        ('B', 'y'): [10, 11, 12]
    })
    result = pd.melt(df, col_level=0)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 7, 8, 9]
    })
    assert_frame_equal(result, expected)

def test_melt_ignore_index():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=[10, 20, 30])
    result = pd.melt(df, ignore_index=False)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    }, index=[10, 10, 10, 20, 20, 20, 30, 30, 30])
    assert_frame_equal(result, expected)

def test_melt_sort():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.melt(df, sort=True)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_verify_integrity():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.melt(df, verify_integrity=True)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_series_concat():
    s = pd.Series([1, 2, 3])
    result = pd.melt(s.to_frame())
    expected = pd.DataFrame({
        'variable': [0, 0, 0],
        'value': [1, 2, 3]
    })
    assert_frame_equal(result, expected)

def test_melt_empty_inputs():
    df = pd.DataFrame()
    result = pd.melt(df)
    assert_frame_equal(result, pd.DataFrame(columns=['variable', 'value']))

def test_melt_mixed_dtypes():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4.0, 5.0, 6.0],
        'C': ['a', 'b', 'c']
    })
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'value': [1, 2, 3, 4.0, 5.0, 6.0, 'a', 'b', 'c']
    })
    assert_frame_equal(result, expected)

def test_melt_non_unique_index():
    df = pd.DataFrame({'A': [1, 2, 3]}, index=[1, 1, 1])
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A'],
        'value': [1, 2, 3]
    })
    assert_frame_equal(result, expected)

def test_melt_multiindex():
    df = pd.DataFrame({
        ('A', 'x'): [1, 2, 3],
        ('A', 'y'): [4, 5, 6],
        ('B', 'x'): [7, 8, 9],
        ('B', 'y'): [10, 11, 12]
    })
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable_0': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'variable_1': ['x', 'x', 'x', 'y', 'x', 'x', 'x', 'y'],
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    })
    assert_frame_equal(result, expected)

def test_melt_datetime():
    df = pd.DataFrame({
        'A': pd.date_range('2022-01-01', periods=3),
        'B': pd.date_range('2022-01-04', periods=3)
    })
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06'])
    })
    assert_frame_equal(result, expected)

def test_melt_categorical():
    df = pd.DataFrame({
        'A': pd.Categorical(['a', 'b', 'c']),
        'B': pd.Categorical(['d', 'e', 'f'])
    })
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': pd.Categorical(['a', 'b', 'c', 'd', 'e', 'f'])
    })
    assert_frame_equal(result, expected)

def test_melt_bool():
    df = pd.DataFrame({
        'A': [True, False, True],
        'B': [False, True, False]
    })
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [True, False, True, False, True, False]
    })
    assert_frame_equal(result, expected)

def test_melt_object():
    df = pd.DataFrame({
        'A': ['a', 'b', 'c'],
        'B': ['d', 'e', 'f']
    })
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': ['a', 'b', 'c', 'd', 'e', 'f']
    })
    assert_frame_equal(result, expected)

def test_melt_numeric():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4.0, 5.0, 6.0]
    })
    result = pd.melt(df)
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 4.0, 5.0, 6.0]
    })
    assert_frame_equal(result, expected)