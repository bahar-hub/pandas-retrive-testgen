# gemini-2.5-pro.py
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

# --- Basic Concatenation Tests ---

def test_concat_series_axis0_default():
    """Test basic concatenation of two Series along axis=0."""
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)
    assert isinstance(result, pd.Series)

def test_concat_dataframes_axis0_default():
    """Test basic concatenation of two DataFrames along axis=0."""
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]],
                            columns=['letter', 'number'], index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_dataframes_axis1():
    """Test concatenation of two DataFrames along axis=1."""
    df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({'B': [3, 4]}, index=[0, 1])
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
    assert_frame_equal(result, expected)

def test_concat_series_axis1():
    """Test concatenation of Series along axis=1, resulting in a DataFrame."""
    s1 = pd.Series([1, 2], name='A')
    s2 = pd.Series([3, 4], name='B')
    result = pd.concat([s1, s2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(result, expected)
    assert isinstance(result, pd.DataFrame)

def test_concat_mixed_series_dataframe():
    """Test concatenating a DataFrame and a Series."""
    df1 = pd.DataFrame({'A': [1, 2]})
    s1 = pd.Series([3, 4], name='B')
    result = pd.concat([df1, s1], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(result, expected)

# --- Parameter: ignore_index ---

def test_concat_axis0_ignore_index_true():
    """Test ignore_index=True for axis=0 concatenation."""
    df1 = pd.DataFrame({'A': [1]}, index=[10])
    df2 = pd.DataFrame({'A': [2]}, index=[20])
    result = pd.concat([df1, df2], ignore_index=True)
    expected = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
    assert_frame_equal(result, expected)

def test_concat_axis1_ignore_index_true():
    """Test ignore_index=True for axis=1 concatenation."""
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'B': [3, 4]})
    result = pd.concat([df1, df2], axis=1, ignore_index=True)
    expected = pd.DataFrame([[1, 3], [2, 4]], columns=[0, 1])
    assert_frame_equal(result, expected)

# --- Parameter: join ---

def test_concat_join_inner():
    """Test inner join on non-concatenation axis."""
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]}, index=[0, 1])
    result = pd.concat([df1, df2], axis=1, join='inner')
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'B': [5, 6], 'C': [7, 8]},
                            index=[0, 1])
    expected.columns = ['A', 'B', 'B', 'C'] # Concat on axis=1 preserves columns
    assert_frame_equal(result, expected)

def test_concat_join_inner_axis0():
    """Test inner join on columns for axis=0 concatenation."""
    df1 = pd.DataFrame({'A': [1], 'B': [2]})
    df2 = pd.DataFrame({'B': [3], 'C': [4]})
    result = pd.concat([df1, df2], join='inner', ignore_index=True)
    expected = pd.DataFrame({'B': [2, 3]})
    assert_frame_equal(result, expected)

def test_concat_join_outer_different_columns():
    """Test outer join (default) with different columns."""
    df1 = pd.DataFrame({'A': [1], 'B': [2]})
    df2 = pd.DataFrame({'B': [3], 'C': [4]})
    result = pd.concat([df1, df2], ignore_index=True)
    expected = pd.DataFrame({'A': [1.0, np.nan], 'B': [2, 3], 'C': [np.nan, 4.0]})
    assert_frame_equal(result, expected)

# --- Parameter: keys, names, levels ---

def test_concat_with_keys_and_names():
    """Test creating a MultiIndex using keys and names."""
    df1 = pd.DataFrame({'A': [1]})
    df2 = pd.DataFrame({'A': [2]})
    result = pd.concat([df1, df2], keys=['one', 'two'], names=['level1', 'level2'])
    expected_index = pd.MultiIndex.from_tuples(
        [('one', 0), ('two', 0)], names=['level1', 'level2']
    )
    expected = pd.DataFrame({'A': [1, 2]}, index=expected_index)
    assert_frame_equal(result, expected)

def test_concat_with_dict_as_objs():
    """Test using a dictionary for objs, which implicitly uses keys."""
    df1 = pd.DataFrame({'A': [1]})
    df2 = pd.DataFrame({'A': [2]})
    objs = {'x': df1, 'y': df2} # Keys are sorted: 'x', 'y'
    result = pd.concat(objs)
    expected_index = pd.MultiIndex.from_tuples([('x', 0), ('y', 0)])
    expected = pd.DataFrame({'A': [1, 2]}, index=expected_index)
    assert_frame_equal(result, expected)

def test_concat_keys_with_ignore_index():
    """Keys should be ignored when ignore_index=True."""
    df1 = pd.DataFrame({'A': [1]})
    df2 = pd.DataFrame({'A': [2]})
    result = pd.concat([df1, df2], keys=['one', 'two'], ignore_index=True)
    expected = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
    assert_frame_equal(result, expected)

def test_concat_with_levels():
    """Test specifying levels for the resulting MultiIndex."""
    s1 = pd.Series([1], index=[0])
    s2 = pd.Series([2], index=[0])
    # Specify a level order and values not present in keys
    result = pd.concat([s1, s2], keys=['x', 'y'], levels=[['y', 'x', 'z']])
    
    # The resulting index should only contain the used keys
    expected_index = pd.MultiIndex.from_tuples([('x', 0), ('y', 0)])
    expected = pd.Series([1, 2], index=expected_index)
    assert_series_equal(result, expected)
    
    # The index's levels attribute should match what was passed
    expected_level = pd.Index(['y', 'x', 'z'])
    assert result.index.levels[0].equals(expected_level)

# --- Parameter: verify_integrity ---

def test_concat_verify_integrity_true_passes():
    """Test verify_integrity=True with unique indices."""
    df1 = pd.DataFrame({'A': [1]}, index=[0])
    df2 = pd.DataFrame({'A': [2]}, index=[1])
    result = pd.concat([df1, df2], verify_integrity=True)
    expected = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
    assert_frame_equal(result, expected)

def test_concat_verify_integrity_true_raises_axis0():
    """Test verify_integrity=True raises ValueError for duplicate indices on axis=0."""
    df1 = pd.DataFrame({'A': [1]}, index=['a'])
    df2 = pd.DataFrame({'A': [2]}, index=['a'])
    with pytest.raises(ValueError, match="Indexes have overlapping values:"):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_verify_integrity_true_raises_axis1():
    """Test verify_integrity=True raises ValueError for duplicate columns on axis=1."""
    df1 = pd.DataFrame([[1]], columns=['A'])
    df2 = pd.DataFrame([[2]], columns=['A'])
    with pytest.raises(ValueError, match="columns have overlapping values:"):
        pd.concat([df1, df2], axis=1, verify_integrity=True)

# --- Parameter: sort ---

def test_concat_sort_true():
    """Test sort=True to sort the non-concatenation axis."""
    df1 = pd.DataFrame({'C': [1], 'A': [2]})
    df2 = pd.DataFrame({'B': [3], 'C': [4]})
    result = pd.concat([df1, df2], sort=True, ignore_index=True)
    expected = pd.DataFrame({'A': [2.0, np.nan], 'B': [np.nan, 3.0], 'C': [1.0, 4.0]})
    assert_frame_equal(result, expected)

def test_concat_sort_false():
    """Test sort=False (default) preserves column order."""
    df1 = pd.DataFrame({'C': [1], 'A': [2]})
    df2 = pd.DataFrame({'B': [3], 'C': [4]})
    result = pd.concat([df1, df2], sort=False, ignore_index=True)
    # Order is determined by the first DataFrame, then the second
    expected = pd.DataFrame({'C': [1.0, 4.0], 'A': [2.0, np.nan], 'B': [np.nan, 3.0]})
    assert_frame_equal(result, expected)

# --- Edge Cases ---

def test_concat_empty_list_raises():
    """Test that concatenating an empty list raises ValueError."""
    with pytest.raises(ValueError, match="No objects to concatenate"):
        pd.concat([])

def test_concat_all_none_raises():
    """Test that concatenating only None objects raises ValueError."""
    with pytest.raises(ValueError, match="No objects to concatenate"):
        pd.concat([None, None])

def test_concat_with_some_none():
    """Test that None objects in the list are silently dropped."""
    df1 = pd.DataFrame({'A': [1]})
    result = pd.concat([df1, None, None])
    assert_frame_equal(result, df1)

def test_concat_empty_dataframes():
    """Test concatenating empty DataFrames."""
    df1 = pd.DataFrame(columns=['A', 'B'])
    df2 = pd.DataFrame(columns=['A', 'B'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame(columns=['A', 'B'], index=pd.Index([], dtype='int64'))
    assert_frame_equal(result, expected)

def test_concat_empty_and_non_empty():
    """Test concatenating an empty DataFrame with a non-empty one."""
    df1 = pd.DataFrame(columns=['A', 'B'])
    df2 = pd.DataFrame({'A': [1], 'B': [2]})
    result = pd.concat([df1, df2], ignore_index=True)
    expected = pd.DataFrame({'A': [1], 'B': [2]})
    assert_frame_equal(result, expected)

def test_concat_non_unique_index():
    """Test concatenation with non-unique indices."""
    df1 = pd.DataFrame({'A': [1, 2]}, index=['a', 'a'])
    df2 = pd.DataFrame({'A': [3, 4]}, index=['b', 'b'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=['a', 'a', 'b', 'b'])
    assert_frame_equal(result, expected)

def test_concat_non_unique_columns_axis1():
    """Test concatenation with non-unique columns on axis=1."""
    df1 = pd.DataFrame([[1, 2]], columns=['A', 'A'])
    df2 = pd.DataFrame([[3, 4]], columns=['B', 'B'])
    result = pd.concat([df1, df2], axis=1, ignore_index=True)
    expected = pd.DataFrame([[1, 2, 3, 4]], columns=[0, 1, 2, 3])
    assert_frame_equal(result, expected)

# --- Data Type Specific Tests ---

def test_concat_categorical_dtype():
    """Test concatenation of Series with categorical dtype."""
    s1 = pd.Series(['a', 'b'], dtype='category')
    s2 = pd.Series(['a', 'c'], dtype='category')
    result = pd.concat([s1, s2], ignore_index=True)
    expected_cats = pd.CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)
    expected = pd.Series(['a', 'b', 'a', 'c'], dtype=expected_cats)
    assert_series_equal(result, expected)

def test_concat_datetime_with_timezone():
    """Test concatenation of Series with the same timezone."""
    tz = 'America/New_York'
    s1 = pd.Series(pd.to_datetime(['2023-01-01']).tz_localize(tz))
    s2 = pd.Series(pd.to_datetime(['2023-01-02']).tz_localize(tz))
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02']).tz_localize(tz))
    assert_series_equal(result, expected)

def test_concat_datetime_mixed_timezone_raises():
    """Test that concatenating different timezones raises TypeError."""
    s1 = pd.Series(pd.to_datetime(['2023-01-01']).tz_localize('UTC'))
    s2 = pd.Series(pd.to_datetime(['2023-01-02']).tz_localize('US/Eastern'))
    with pytest.raises(TypeError):
        pd.concat([s1, s2])

def test_concat_upcast_int_to_float():
    """Test dtype upcasting from int to float."""
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3.5, 4.5])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series([1.0, 2.0, 3.5, 4.5])
    assert_series_equal(result, expected)
    assert result.dtype == 'float64'

def test_concat_upcast_to_object():
    """Test dtype upcasting to object."""
    s1 = pd.Series([1, 2])
    s2 = pd.Series(['a', 'b'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series([1, 2, 'a', 'b'], dtype='object')
    assert_series_equal(result, expected)

def test_concat_boolean_and_numeric():
    """Test concatenation of boolean and integer types."""
    s1 = pd.Series([True, False])
    s2 = pd.Series([0, 1, 5])
    result = pd.concat([s1, s2], ignore_index=True)
    # Booleans are coerced to integers
    expected = pd.Series([1, 0, 0, 1, 5], dtype='int64')
    assert_series_equal(result, expected)

# --- MultiIndex Tests ---

def test_concat_multiindex_rows_axis0():
    """Test concatenation of DataFrames with MultiIndex on rows."""
    idx1 = pd.MultiIndex.from_tuples([('a', 1), ('a', 2)])
    df1 = pd.DataFrame({'X': [10, 20]}, index=idx1)
    idx2 = pd.MultiIndex.from_tuples([('b', 1), ('b', 2)])
    df2 = pd.DataFrame({'X': [30, 40]}, index=idx2)
    result = pd.concat([df1, df2])
    expected_idx = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
    expected = pd.DataFrame({'X': [10, 20, 30, 40]}, index=expected_idx)
    assert_frame_equal(result, expected)

def test_concat_multiindex_columns_axis1():
    """Test concatenation of DataFrames with MultiIndex on columns."""
    cols1 = pd.MultiIndex.from_tuples([('A', 'cat')])
    df1 = pd.DataFrame([[1], [2]], columns=cols1)
    cols2 = pd.MultiIndex.from_tuples([('B', 'dog')])
    df2 = pd.DataFrame([[3], [4]], columns=cols2)
    result = pd.concat([df1, df2], axis=1)
    expected_cols = pd.MultiIndex.from_tuples([('A', 'cat'), ('B', 'dog')])
    expected = pd.DataFrame([[1, 3], [2, 4]], columns=expected_cols)
    assert_frame_equal(result, expected)

# --- Other Parameter Tests ---

def test_concat_axis_string_aliases():
    """Test using 'index' and 'columns' as string aliases for axis."""
    df1 = pd.DataFrame({'A': [1]})
    df2 = pd.DataFrame({'A': [2]})
    df3 = pd.DataFrame({'B': [3]})
    
    result_index = pd.concat([df1, df2], axis='index')
    expected_index = pd.concat([df1, df2], axis=0)
    assert_frame_equal(result_index, expected_index)
    
    result_columns = pd.concat([df1, df3], axis='columns')
    expected_columns = pd.concat([df1, df3], axis=1)
    assert_frame_equal(result_columns, expected_columns)

@pytest.mark.skipif(
    pd.options.mode.copy_on_write,
    reason="Behavior of copy=False differs with Copy-on-Write"
)
def test_copy_false_allows_view_modification():
    """
    Test that with copy=False, a modification to the source may affect the result.
    This test is sensitive to internal implementation and pandas version.
    It is skipped if Copy-on-Write is enabled, as CoW changes this behavior.
    """
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    
    # With copy=False, concat may not copy the underlying numpy array
    result = pd.concat([df1, df2], copy=False, ignore_index=True)
    
    # Modify the original DataFrame
    df1.iloc[0, 0] = 99
    
    # Check if the concatenated result is also modified
    # This indicates a view was created, not a copy
    assert result.iloc[0, 0] == 99