import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
from typing import List, Tuple

# Test basic Series concatenation (axis=0)
def test_basic_series_concat_axis0():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)

def test_basic_series_concat_ignore_index():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series(['a', 'b', 'c', 'd'])
    assert_series_equal(result, expected)

def test_series_concat_keys():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    expected = pd.Series(['a', 'b', 'c', 'd'], 
                        index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)]))
    assert_series_equal(result, expected)

def test_series_concat_names():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series', 'Row'])
    expected_index = pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)], 
                                              names=['Series', 'Row'])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

# Test DataFrame concatenation (axis=0)
def test_basic_dataframe_concat_axis0():
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], 
                           columns=['letter', 'number'], 
                           index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_dataframe_concat_different_columns_outer():
    df1 = pd.DataFrame([['a', 1]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['c', 3, 'cat']], columns=['letter', 'number', 'animal'])
    result = pd.concat([df1, df2], sort=False)
    expected = pd.DataFrame([
        ['a', 1, np.nan],
        ['c', 3, 'cat']
    ], columns=['letter', 'number', 'animal'], index=[0, 0])
    assert_frame_equal(result, expected)

def test_dataframe_concat_different_columns_inner():
    df1 = pd.DataFrame([['a', 1]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['c', 3, 'cat']], columns=['letter', 'number', 'animal'])
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame([
        ['a', 1],
        ['c', 3]
    ], columns=['letter', 'number'], index=[0, 0])
    assert_frame_equal(result, expected)

# Test axis=1 concatenation
def test_dataframe_concat_axis1():
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df4 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']], 
                      columns=['animal', 'name'])
    result = pd.concat([df1, df4], axis=1)
    expected = pd.DataFrame([
        ['a', 1, 'bird', 'polly'],
        ['b', 2, 'monkey', 'george']
    ], columns=['letter', 'number', 'animal', 'name'])
    assert_frame_equal(result, expected)

def test_series_to_dataframe_concat_axis1():
    s1 = pd.Series(['x', 'y'])
    df1 = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    result = pd.concat([df1, s1], axis=1)
    expected = pd.DataFrame([
        [1, 2, 'x'],
        [3, 4, 'y']
    ], columns=['A', 'B', 0])
    assert_frame_equal(result, expected)

# Test empty objects
def test_concat_empty_list():
    result = pd.concat([])
    expected = pd.DataFrame()
    assert_frame_equal(result, expected)

def test_concat_empty_dataframe():
    df1 = pd.DataFrame({'A': [1, 2]})
    result = pd.concat([df1, pd.DataFrame()])
    expected = pd.DataFrame({'A': [1, 2]})
    assert_frame_equal(result, expected)

def test_concat_all_empty():
    with pytest.raises(ValueError, match="No objects to concatenate"):
        pd.concat([pd.DataFrame(), pd.Series()])

# Test verify_integrity
def test_verify_integrity_duplicate_index():
    df5 = pd.DataFrame([1], index=['a'])
    df6 = pd.DataFrame([2], index=['a'])
    with pytest.raises(ValueError, match="Indexes have overlapping values"):
        pd.concat([df5, df6], verify_integrity=True)

def test_verify_integrity_no_duplicates():
    df1 = pd.DataFrame({'A': [1]}, index=[0])
    df2 = pd.DataFrame({'A': [2]}, index=[1])
    result = pd.concat([df1, df2], verify_integrity=True)
    expected = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
    assert_frame_equal(result, expected)

# Test mixed dtypes
def test_mixed_dtypes():
    s1 = pd.Series([1, 2])
    s2 = pd.Series(['a', 'b'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series([1, 2, 'a', 'b'])
    assert_series_equal(result, expected)

def test_numeric_boolean_mix():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [True, False]})
    df2 = pd.DataFrame({'A': [3.0, 4.0], 'B': [True, True]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({
        'A': [1, 2, 3.0, 4.0],
        'B': [True, False, True, True]
    }, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

# Test categorical dtypes
def test_categorical_concat():
    cat1 = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    cat2 = pd.Categorical(['b', 'c'], categories=['a', 'b', 'c'])
    s1 = pd.Series(cat1)
    s2 = pd.Series(cat2)
    result = pd.concat([s1, s2])
    expected = pd.Series(pd.Categorical(['a', 'b', 'b', 'c'], 
                                       categories=['a', 'b', 'c']))
    assert_series_equal(result, expected)

# Test datetime and timezone
def test_datetime_concat():
    df1 = pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-01-02'])})
    df2 = pd.DataFrame({'date': pd.to_datetime(['2023-01-03', '2023-01-04'])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    }, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_timezone_concat():
    tz = 'US/Eastern'
    df1 = pd.DataFrame({'dt': pd.date_range('2023-01-01', periods=2, tz=tz)})
    df2 = pd.DataFrame({'dt': pd.date_range('2023-01-03', periods=2, tz=tz)})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({
        'dt': pd.date_range('2023-01-01', '2023-01-04', freq='D', tz=tz)
    }, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

# Test MultiIndex
def test_multiindex_concat_axis0():
    midx1 = pd.MultiIndex.from_tuples([('x', 1), ('x', 2)], names=['level0', 'level1'])
    midx2 = pd.MultiIndex.from_tuples([('y', 1), ('y', 2)], names=['level0', 'level1'])
    df1 = pd.DataFrame({'A': [1, 2]}, index=midx1)
    df2 = pd.DataFrame({'A': [3, 4]}, index=midx2)
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, 
                           index=midx1.append(midx2).rename(names=['level0', 'level1']))
    assert_frame_equal(result, expected)

def test_multiindex_concat_axis1():
    tuples = [('a', 1), ('a', 2), ('b', 1)]
    midx = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    df1 = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1], columns=midx[:2])
    df2 = pd.DataFrame([[5, 6]], index=[0], columns=[midx[2]])
    result = pd.concat([df1, df2], axis=1)
    expected_columns = midx[:2].append(midx[2:])
    expected = pd.DataFrame([[1, 2, 5], [3, 4, np.nan]], 
                           index=[0, 1], columns=expected_columns)
    assert_frame_equal(result, expected)

# Test sort parameter
def test_sort_false():
    df1 = pd.DataFrame({'B': [1], 'A': [2]}, index=[0])
    df2 = pd.DataFrame({'A': [3], 'B': [4]}, index=[0])
    result = pd.concat([df1, df2], sort=False)
    expected = pd.DataFrame([
        [2, 1],
        [3, 4]
    ], columns=['B', 'A', 'B', 'A'], index=[0, 0])
    assert_frame_equal(result, expected)

def test_sort_true():
    df1 = pd.DataFrame({'B': [1], 'A': [2]}, index=[0])
    df2 = pd.DataFrame({'A': [3], 'B': [4]}, index=[0])
    result = pd.concat([df1, df2], sort=True)
    expected_columns = pd.Index(['A', 'B'], name=None)
    expected = pd.DataFrame([
        [2, 1],
        [3, 4]
    ], columns=expected_columns.repeat(2), index=[0, 0])
    assert_frame_equal(result, expected)

# Test non-unique indices/columns
def test_duplicate_columns():
    df1 = pd.DataFrame([[1, 2]], columns=['A', 'A'])
    df2 = pd.DataFrame([[3, 4]], columns=['A', 'B'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame([
        [1, 2, np.nan],
        [3, 4, np.nan]
    ], columns=['A', 'A', 'B'], index=[0, 0])
    assert_frame_equal(result, expected)

# Test mapping input
def test_mapping_input():
    mapping = {'df1': pd.DataFrame({'A': [1]}), 'df2': pd.DataFrame({'A': [2]})}
    result = pd.concat(mapping)
    expected = pd.DataFrame({'A': [1, 2]}, index=[0, 0])
    assert_frame_equal(result, expected)

def test_mapping_with_keys():
    mapping = {'df1': pd.DataFrame({'A': [1]}), 'df2': pd.DataFrame({'A': [2]})}
    result = pd.concat(mapping, keys=['k1', 'k2'])
    expected_index = pd.MultiIndex.from_tuples([('k1', 'df1', 0), ('k2', 'df2', 0)], 
                                              names=['keys', None, None])
    expected = pd.DataFrame({'A': [1, 2]}, index=expected_index)
    assert_frame_equal(result, expected)

# Test levels parameter
def test_levels_parameter():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['one', 'two'], levels=[['one', 'two', 'three']])
    expected_index = pd.MultiIndex.from_tuples([('one', 0), ('one', 1), ('two', 0), ('two', 1)], 
                                              levels=[['one', 'two', 'three'], None],
                                              names=[None, None])
    expected = pd.Series([1, 2, 3, 4], index=expected_index)
    assert_series_equal(result, expected)

# Test copy parameter (observable via mutation)
def test_copy_false_mutation():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], copy=False)
    df1.loc[0, 'A'] = 99
    # If copy=False worked, original shouldn't affect result
    expected = pd.DataFrame({'A': [99, 2, 3, 4]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

# Test single Series returns Series
def test_single_series_returns_series():
    s1 = pd.Series([1, 2, 3])
    result = pd.concat([s1])
    assert_series_equal(result, s1)

# Test Series with name
def test_series_with_name():
    s1 = pd.Series([1, 2], name='A')
    s2 = pd.Series([3, 4], name='A')
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 0, 1], name='A')
    assert_series_equal(result, expected)