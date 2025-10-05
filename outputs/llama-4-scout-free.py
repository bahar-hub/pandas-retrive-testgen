import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

def test_concat_series():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4])
    assert_series_equal(result, expected)

def test_concat_dataframe():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]})
    assert_frame_equal(result, expected)

def test_concat_axis_1():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'B': [3, 4]})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(result, expected)

def test_concat_ignore_index():
    df1 = pd.DataFrame({'A': [1, 2]}, index=[2, 3])
    df2 = pd.DataFrame({'A': [3, 4]}, index=[4, 5])
    result = pd.concat([df1, df2], ignore_index=True)
    expected = pd.DataFrame({'A': [1, 2, 3, 4]})
    assert_frame_equal(result, expected)

def test_concat_keys():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)]))
    assert_series_equal(result, expected)

def test_concat_levels():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['s1', 's2'], levels=[['s1', 's2'], [0, 1]])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)]))
    assert_series_equal(result, expected)

def test_concat_names():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series name', 'Row ID'])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)], names=['Series name', 'Row ID']))
    assert_series_equal(result, expected)

def test_concat_verify_integrity():
    df1 = pd.DataFrame({'A': [1]}, index=[0])
    df2 = pd.DataFrame({'A': [2]}, index=[0])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_sort():
    df1 = pd.DataFrame({'A': [1, 2]}, index=[2, 1])
    df2 = pd.DataFrame({'A': [3, 4]}, index=[4, 3])
    result = pd.concat([df1, df2], sort=True)
    expected = pd.DataFrame({'A': [2, 1, 4, 3]})
    assert_frame_equal(result, expected)

def test_concat_join_inner():
    df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 2])
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'A': [2, 3]})
    assert_frame_equal(result, expected)

def test_concat_empty():
    result = pd.concat([])
    expected = pd.DataFrame()
    assert_frame_equal(result, expected)

def test_concat_mixed_dtype():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'B': ['a', 'b']})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': ['a', 'b']})
    assert_frame_equal(result, expected)

def test_concat_non_unique_index():
    df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 0])
    df2 = pd.DataFrame({'A': [3, 4]}, index=[0, 1])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=[0, 0, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_multi_index():
    df1 = pd.DataFrame({'A': [1, 2]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1)]))
    df2 = pd.DataFrame({'A': [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 2), (0, 3)]))
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (0, 2), (0, 3)]))
    assert_frame_equal(result, expected)

def test_concat_categorical_dtype():
    df1 = pd.DataFrame({'A': pd.Series(['a', 'b'], dtype='category')})
    df2 = pd.DataFrame({'A': pd.Series(['c', 'd'], dtype='category')})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': pd.Series(['a', 'b', 'c', 'd'], dtype='category')})
    assert_frame_equal(result, expected)

def test_concat_datetime_dtype():
    df1 = pd.DataFrame({'A': pd.Series([pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02')])})
    df2 = pd.DataFrame({'A': pd.Series([pd.Timestamp('2022-01-03'), pd.Timestamp('2022-01-04')])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': pd.Series([pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02'), pd.Timestamp('2022-01-03'), pd.Timestamp('2022-01-04')])})
    assert_frame_equal(result, expected)

def test_concat_copy():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], copy=False)
    expected = pd.DataFrame({'A': [1, 2, 3, 4]})
    assert_frame_equal(result, expected)