import pytest
import pandas as pd
import numpy as np

def test_concat_series():
    s1 = pd.Series([1, 2], name='s1')
    s2 = pd.Series([3, 4], name='s2')
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4], name='s1')
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_ignore_index():
    s1 = pd.Series([1, 2], name='s1')
    s2 = pd.Series([3, 4], name='s2')
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series([1, 2, 3, 4], index=pd.RangeIndex(4))
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_keys():
    s1 = pd.Series([1, 2], name='s1')
    s2 = pd.Series([3, 4], name='s2')
    result = pd.concat([s1, s2], keys=['a', 'b'])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('a', 0), ('a', 1), ('b', 0), ('b', 1)]))
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_names():
    s1 = pd.Series([1, 2], name='s1')
    s2 = pd.Series([3, 4], name='s2')
    result = pd.concat([s1, s2], keys=['a', 'b'], names=['key', 'index'])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('a', 0), ('a', 1), ('b', 0), ('b', 1)]), name='s1')
    expected.index.names = ['key', 'index']
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
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]}, index=pd.RangeIndex(4))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_join_inner():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'a': [1, 2, 5, 6]}, index=pd.RangeIndex(4))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_axis_1():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'c': [5, 6], 'd': [7, 8]})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_empty_dataframes():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    result = pd.concat([df1, df2])
    expected = pd.DataFrame()
    pd.testing.assert_frame_equal(result, expected)

def test_concat_mixed_dtypes():
    df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    df2 = pd.DataFrame({'a': [3, 4], 'b': [5.0, 6.0]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['x', 'y', 5.0, 6.0]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_non_unique_indices():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 0])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[0, 0])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=[0, 0, 0, 0])
    pd.testing.assert_frame_equal(result, expected)

def test_concat_multiindex_rows():
    df1 = pd.DataFrame({'a': [1, 2]}, index=pd.MultiIndex.from_tuples([('x', 0), ('x', 1)]))
    df2 = pd.DataFrame({'a': [3, 4]}, index=pd.MultiIndex.from_tuples([('y', 0), ('y', 1)]))
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=pd.MultiIndex.from_tuples([('x', 0), ('x', 1), ('y', 0), ('y', 1)]))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_multiindex_columns():
    df1 = pd.DataFrame({'a': [1, 2]}, columns=pd.MultiIndex.from_tuples([('x', 0)]))
    df2 = pd.DataFrame({'a': [3, 4]}, columns=pd.MultiIndex.from_tuples([('y', 0)]))
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'a': [1, 3]}, columns=pd.MultiIndex.from_tuples([('x', 0), ('y', 0)]))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_different_columns():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, np.nan, np.nan], 'c': [np.nan, np.nan, 7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_categorical_dtypes():
    df1 = pd.DataFrame({'a': pd.Categorical(['x', 'y'])})
    df2 = pd.DataFrame({'a': pd.Categorical(['z', 'w'])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': pd.Categorical(['x', 'y', 'z', 'w'])})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_datetime_dtypes():
    df1 = pd.DataFrame({'a': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    df2 = pd.DataFrame({'a': pd.to_datetime(['2020-01-03', '2020-01-04'])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_boolean_mix():
    df1 = pd.DataFrame({'a': [True, False], 'b': [1, 2]})
    df2 = pd.DataFrame({'a': [False, True], 'b': [3, 4]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [True, False, False, True], 'b': [1, 2, 3, 4]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_object_mix():
    df1 = pd.DataFrame({'a': ['x', 'y'], 'b': [1, 2]})
    df2 = pd.DataFrame({'a': [3, 4], 'b': [5.0, 6.0]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': ['x', 'y', 3, 4], 'b': [1, 2, 5.0, 6.0]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_numeric_mix():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3.0, 4.0, 7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_verify_integrity():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 0])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[0, 0])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_sort():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[2, 1])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[4, 3])
    result = pd.concat([df1, df2], sort=True)
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=[1, 2, 3, 4])
    pd.testing.assert_frame_equal(result, expected)

def test_concat_copy_false():
    df1 = pd.DataFrame({'a': [1, 2]})
    df2 = pd.DataFrame({'a': [3, 4]})
    result = pd.concat([df1, df2], copy=False)
    # Verify that the result is a view, not a copy
    df1['a'][0] = 10
    expected = pd.DataFrame({'a': [10, 2, 3, 4]})
    pd.testing.assert_frame_equal(result, expected)