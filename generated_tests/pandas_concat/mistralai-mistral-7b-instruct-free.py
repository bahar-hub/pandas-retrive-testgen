import pytest
import pandas as pd
import numpy as np

def test_concat_series():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = pd.Series(['a', 'b', 'c', 'd'])
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_ignore_index():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series(['a', 'b', 'c', 'd'], index=[0, 1, 2, 3])
    pd.testing.assert_series_equal(result, expected)

def test_concat_series_keys():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=pd.Index(['s1', 's1', 's2', 's2'], name='keys'))
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
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]}, index=[0, 1, 2, 3])
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_keys():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    result = pd.concat([df1, df2], keys=['df1', 'df2'])
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]})
    expected.index = pd.MultiIndex.from_tuples([('df1', 0), ('df1', 1), ('df2', 0), ('df2', 1)], names=['keys', None])
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_join_inner():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'a': [1, 2, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_join_outer():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})
    result = pd.concat([df1, df2], join='outer')
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, np.nan, np.nan], 'c': [np.nan, np.nan, 7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_dataframes_axis_1():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'c': [5, 6], 'd': [7, 8]})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_empty_dataframe():
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
    df1 = pd.DataFrame({'a': [1, 2]}, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)]))
    df2 = pd.DataFrame({'a': [3, 4]}, index=pd.MultiIndex.from_tuples([('y', 1), ('y', 2)]))
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)]))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_multiindex_columns():
    df1 = pd.DataFrame({'a': [1, 2]}, columns=pd.MultiIndex.from_tuples([('x', 1)]))
    df2 = pd.DataFrame({'a': [3, 4]}, columns=pd.MultiIndex.from_tuples([('y', 1)]))
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, columns=pd.MultiIndex.from_tuples([('x', 1), ('y', 1)]))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_categorical_dtypes():
    df1 = pd.DataFrame({'a': pd.Categorical(['x', 'y'])})
    df2 = pd.DataFrame({'a': pd.Categorical(['y', 'z'])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': pd.Categorical(['x', 'y', 'y', 'z'])})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_datetime_dtypes():
    df1 = pd.DataFrame({'a': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    df2 = pd.DataFrame({'a': pd.to_datetime(['2020-01-03', '2020-01-04'])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_boolean_mix():
    df1 = pd.DataFrame({'a': [True, False]})
    df2 = pd.DataFrame({'a': [1, 2]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'a': [True, False, 1, 2]})
    pd.testing.assert_frame_equal(result, expected)

def test_concat_verify_integrity():
    df1 = pd.DataFrame({'a': [1]}, index=['x'])
    df2 = pd.DataFrame({'a': [2]}, index=['x'])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_sort():
    df1 = pd.DataFrame({'a': [1, 2]}, index=['b', 'a'])
    df2 = pd.DataFrame({'a': [3, 4]}, index=['d', 'c'])
    result = pd.concat([df1, df2], sort=True)
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
    pd.testing.assert_frame_equal(result, expected)