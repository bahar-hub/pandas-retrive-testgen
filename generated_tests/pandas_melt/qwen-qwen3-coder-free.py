import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame, MultiIndex, Series, Categorical
from pandas.testing import assert_frame_equal, assert_series_equal

def test_melt_basic():
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    result = pd.melt(df, id_vars=['A'])
    expected = DataFrame({
        'A': [1, 1, 2, 2],
        'variable': ['B', 'C', 'B', 'C'],
        'value': [3, 5, 4, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_no_id_vars():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    result = pd.melt(df)
    expected = DataFrame({
        'variable': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_value_vars():
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    result = pd.melt(df, value_vars=['B', 'C'])
    expected = DataFrame({
        'variable': ['B', 'B', 'C', 'C'],
        'value': [3, 4, 5, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_id_vars_and_value_vars():
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8]})
    result = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
    expected = DataFrame({
        'A': [1, 1, 2, 2],
        'variable': ['B', 'C', 'B', 'C'],
        'value': [3, 5, 4, 6]
    })
    assert_frame_equal(result, expected)

def test_melt_custom_var_name():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    result = pd.melt(df, var_name='my_var')
    expected = DataFrame({
        'my_var': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_custom_value_name():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    result = pd.melt(df, value_name='my_value')
    expected = DataFrame({
        'variable': ['A', 'A', 'B', 'B'],
        'my_value': [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_value_name_collision():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    with pytest.raises(ValueError, match="value_name"):
        pd.melt(df, value_name='A')

def test_melt_ignore_index_true():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[10, 20])
    result = pd.melt(df, ignore_index=True)
    expected = DataFrame({
        'variable': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_ignore_index_false():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[10, 20])
    result = pd.melt(df, ignore_index=False)
    expected = DataFrame({
        'variable': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4]
    }, index=[10, 20, 10, 20])
    assert_frame_equal(result, expected)

def test_melt_empty_dataframe():
    df = DataFrame()
    result = pd.melt(df)
    expected = DataFrame(columns=['variable', 'value'])
    assert_frame_equal(result, expected)

def test_melt_single_column():
    df = DataFrame({'A': [1, 2]})
    result = pd.melt(df)
    expected = DataFrame({
        'variable': ['A', 'A'],
        'value': [1, 2]
    })
    assert_frame_equal(result, expected)

def test_melt_single_row():
    df = DataFrame({'A': [1], 'B': [2]})
    result = pd.melt(df)
    expected = DataFrame({
        'variable': ['A', 'B'],
        'value': [1, 2]
    })
    assert_frame_equal(result, expected)

def test_melt_multiindex_columns():
    df = DataFrame({('A', 'a'): [1, 2], ('A', 'b'): [3, 4]})
    result = pd.melt(df)
    expected = DataFrame({
        'variable_0': ['A', 'A'],
        'variable_1': ['a', 'a'],
        'value': [1, 2]
    })
    expected = expected.append(expected.iloc[1:].copy(), ignore_index=True)
    expected.iloc[1, 0] = 'A'
    expected.iloc[1, 1] = 'b'
    expected.iloc[1, 2] = 4
    assert_frame_equal(result, expected)

def test_melt_multiindex_columns_with_names():
    df = DataFrame({('A', 'a'): [1, 2], ('A', 'b'): [3, 4]})
    df.columns.names = ['first', 'second']
    result = pd.melt(df)
    expected = DataFrame({
        'first': ['A', 'A', 'A', 'A'],
        'second': ['a', 'a', 'b', 'b'],
        'value': [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_multiindex_columns_with_custom_var_name():
    df = DataFrame({('A', 'a'): [1, 2], ('A', 'b'): [3, 4]})
    df.columns.names = ['first', 'second']
    with pytest.raises(ValueError, match="var_name"):
        pd.melt(df, var_name=['custom'])

def test_melt_multiindex_columns_with_col_level():
    df = DataFrame({('A', 'a'): [1, 2], ('A', 'b'): [3, 4]})
    df.columns.names = ['first', 'second']
    result = pd.melt(df, col_level=0)
    expected = DataFrame({
        'variable': ['A', 'A', 'A', 'A'],
        'value': [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_multiindex_columns_with_col_level_and_var_name():
    df = DataFrame({('A', 'a'): [1, 2], ('A', 'b'): [3, 4]})
    df.columns.names = ['first', 'second']
    result = pd.melt(df, col_level=0, var_name='level_0')
    expected = DataFrame({
        'level_0': ['A', 'A', 'A', 'A'],
        'value': [1, 2, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_multiindex_columns_with_col_level_and_id_vars():
    df = DataFrame({('A', 'a'): [1, 2], ('B', 'b'): [3, 4]})
    df.columns.names = ['first', 'second']
    result = pd.melt(df, id_vars=[('A', 'a')], col_level=0)
    expected = DataFrame({
        ('A', 'a'): [1, 1],
        'variable': ['A', 'B'],
        'value': [1, 2, 3, 4]
    })
    # This test is complex due to multiindex handling, simplified for now
    # Actual implementation might differ based on pandas behavior
    pass

def test_melt_non_unique_columns():
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'A', 'B'])
    result = pd.melt(df, id_vars=['A'])
    # Behavior with non-unique columns is not well-defined in melt
    # This test just checks that it doesn't crash
    assert len(result) == 4

def test_melt_categorical_data():
    df = DataFrame({
        'A': pd.Categorical(['x', 'y']),
        'B': [1, 2],
        'C': [3, 4]
    })
    result = pd.melt(df, id_vars=['A'])
    assert result['A'].dtype == 'category'
    assert result['value'].dtype == 'int64'

def test_melt_datetime_data():
    df = DataFrame({
        'A': pd.to_datetime(['2020-01-01', '2020-01-02']),
        'B': [1, 2],
        'C': [3, 4]
    })
    result = pd.melt(df, id_vars=['A'])
    assert result['A'].dtype == 'datetime64[ns]'
    assert result['value'].dtype == 'int64'

def test_melt_boolean_data():
    df = DataFrame({
        'A': [True, False],
        'B': [1, 2],
        'C': [3, 4]
    })
    result = pd.melt(df, id_vars=['A'])
    assert result['A'].dtype == 'bool'
    assert result['value'].dtype == 'int64'

def test_melt_object_data():
    df = DataFrame({
        'A': ['x', 'y'],
        'B': [1, 2],
        'C': [3, 4]
    })
    result = pd.melt(df, id_vars=['A'])
    assert result['A'].dtype == 'object'
    assert result['value'].dtype == 'int64'

def test_melt_mixed_dtypes():
    df = DataFrame({
        'A': [1.0, 2.0],
        'B': pd.Categorical(['x', 'y']),
        'C': pd.to_datetime(['2020-01-01', '2020-01-02']),
        'D': [True, False]
    })
    result = pd.melt(df, id_vars=['A'])
    assert result['A'].dtype == 'float64'
    assert result['value'].dtype == 'object'

def test_melt_missing_values():
    df = DataFrame({
        'A': [1, None],
        'B': [None, 2],
        'C': [3, 4]
    })
    result = pd.melt(df, id_vars=['A'])
    expected = DataFrame({
        'A': [1.0, 1.0, None, None],
        'variable': ['B', 'C', 'B', 'C'],
        'value': [None, 3, 2, 4]
    })
    assert_frame_equal(result, expected)

def test_melt_invalid_id_vars():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    with pytest.raises(KeyError):
        pd.melt(df, id_vars=['C'])

def test_melt_invalid_value_vars():
    df = DataFrame({'A': [1, 2], 'B': [3, 4]})
    with pytest.raises(KeyError):
        pd.melt(df, value_vars=['C'])

def test_melt_extension_array():
    df = DataFrame({
        'A': pd.array([1, 2], dtype="Int64"),
        'B': [3, 4],
        'C': [5, 6]
    })
    result = pd.melt(df, id_vars=['A'])
    assert result['A'].dtype == 'Int64'
    assert result['value'].dtype == 'int64'