import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

def test_melt_basic():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 4, 5, 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=['A', 'B', 'C'], var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_id_vars():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'A': [1, 1, 2, 2, 3, 3],
        'variable': ['B', 'C', 'B', 'C', 'B', 'C'],
        'value': [4, 7, 5, 8, 6, 9]
    })
    result = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'], var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_all_columns():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 4, 5, 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_empty_dataframe():
    df = pd.DataFrame()
    expected = pd.DataFrame(columns=['variable', 'value'])
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_mixed_dtypes():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z'],
        'C': [True, False, True]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 'x', 'y', True, False]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_multiindex():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    df.columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C')])
    expected = pd.DataFrame({
        'variable': [('X', 'A'), ('X', 'A'), ('X', 'B'), ('X', 'B'), ('Y', 'C'), ('Y', 'C')],
        'value': [1, 2, 4, 5, 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_col_level():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    df.columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C')])
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 4, 5, 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value', col_level=1)
    assert_frame_equal(result, expected)

def test_melt_with_ignore_index_false():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 4, 5, 7, 8]
    })
    expected.index = pd.Index([0, 0, 0, 0, 0, 0])
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value', ignore_index=False)
    assert_frame_equal(result, expected)

def test_melt_with_value_name_conflict():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'value': [7, 8, 9]
    })
    with pytest.raises(ValueError):
        pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')

def test_melt_with_missing_columns():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    with pytest.raises(KeyError):
        pd.melt(df, id_vars=['A'], value_vars=['C', 'D'], var_name='variable', value_name='value')

def test_melt_with_series_input():
    s = pd.Series([1, 2, 3], name='A')
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'A'],
        'value': [1, 2, 3]
    })
    result = pd.melt(s, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_datetime():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, pd.Timestamp('2021-01-01'), pd.Timestamp('2021-01-02'), 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_categorical():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': pd.Categorical(['x', 'y', 'z']),
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, pd.Categorical(['x', 'y', 'z'])[0], pd.Categorical(['x', 'y', 'z'])[1], 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_boolean():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [True, False, True],
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, True, False, 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_object():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z'],
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 'x', 'y', 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)

def test_melt_with_numeric():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4.1, 5.2, 6.3],
        'C': [7, 8, 9]
    })
    expected = pd.DataFrame({
        'variable': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 4.1, 5.2, 7, 8]
    })
    result = pd.melt(df, id_vars=[], value_vars=None, var_name='variable', value_name='value')
    assert_frame_equal(result, expected)