import pandas as pd
import numpy as np
import pytest

def test_concat_series():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=[0, 1, 0, 1])
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_ignore_index():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series(['a', 'b', 'c', 'd'])
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_keys():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)]))
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_names():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series name', 'Row ID'])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)], names=['Series name', 'Row ID']))
    pd.testing.assert_series_equal(result, expected)

def test_concat_dataframes():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_ignore_index():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    result = pd.concat([df1, df2], ignore_index=True)
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_keys():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    result = pd.concat([df1, df2], keys=['df1', 'df2'])
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]}, index=pd.MultiIndex.from_tuples([('df1', 0), ('df1', 1), ('df2', 0), ('df2', 1)]))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_names():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    result = pd.concat([df1, df2], keys=['df1', 'df2'], names=['DataFrame name', 'Row ID'])
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]}, index=pd.MultiIndex.from_tuples([('df1', 0), ('df1', 1), ('df2', 0), ('df2', 1)], names=['DataFrame name', 'Row ID']))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_axis_1():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'c': [5, 6], 'd': [7, 8]})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_join_inner():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'a': [1, 2, 5, 6]}, index=[0, 1, 0, 1])
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_verify_integrity():
    df1 = pd.DataFrame({'a': [1, 2]}, index=['x', 'y'])
    df2 = pd.DataFrame({'a': [3, 4]}, index=['y', 'z'])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_empty_series():
    s1 = pd.Series([])
    s2 = pd.Series([])
    result = pd.concat([s1, s2])
    expected = pd.Series([], index=pd.Index([], dtype='int64'))
    pd.testing.assert_series_equal(result, expected)

def test_concat_empty_dataframes():
    df1 = pd.DataFrame({'a': []})
    df2 = pd.DataFrame({'a': []})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': []})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_mixed_dtypes():
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series(['a', 'b', 'c'])
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 'a', 'b', 'c'], index=[0, 1, 2, 0, 1, 2])
    pd.testing.assert_series_equal(result, expected)

def test_concat_categorical_dtypes():
    s1 = pd.Series([1, 2, 3], dtype='category')
    s2 = pd.Series([4, 5, 6], dtype='category')
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4, 5, 6], index=[0, 1, 2, 0, 1, 2], dtype='category')
    pd.testing.assert_series_equal(result, expected)

def test_concat_datetime_dtypes():
    s1 = pd.Series([pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02')])
    s2 = pd.Series([pd.Timestamp('2022-01-03'), pd.Timestamp('2022-01-04')])
    result = pd.concat([s1, s2])
    expected = pd.Series([pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02'), pd.Timestamp('2022-01-03'), pd.Timestamp('2022-01-04')], index=[0, 1, 0, 1])
    pd.testing.assert_series_equal(result, expected)

def test_concat_timezone_dtypes():
    s1 = pd.Series([pd.Timestamp('2022-01-01', tz='UTC'), pd.Timestamp('2022-01-02', tz='UTC')])
    s2 = pd.Series([pd.Timestamp('2022-01-03', tz='UTC'), pd.Timestamp('2022-01-04', tz='UTC')])
    result = pd.concat([s1, s2])
    expected = pd.Series([pd.Timestamp('2022-01-01', tz='UTC'), pd.Timestamp('2022-01-02', tz='UTC'), pd.Timestamp('2022-01-03', tz='UTC'), pd.Timestamp('2022-01-04', tz='UTC')], index=[0, 1, 0, 1])
    pd.testing.assert_series_equal(result, expected)

def test_concat_boolean_dtypes():
    s1 = pd.Series([True, False])
    s2 = pd.Series([True, False])
    result = pd.concat([s1, s2])
    expected = pd.Series([True, False, True, False], index=[0, 1, 0, 1])
    pd.testing.assert_series_equal(result, expected)

def test_concat_object_dtypes():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=[0, 1, 0, 1])
    pd.testing.assert_series_equal(result, expected)

def test_concat_numeric_dtypes():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 0, 1])
    pd.testing.assert_series_equal(result, expected)