import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

def test_concat_series_axis0():
    s1 = pd.Series([1, 2], name='s1')
    s2 = pd.Series([3, 4], name='s2')
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4], name='s1')
    assert_series_equal(result, expected)

def test_concat_series_axis1():
    s1 = pd.Series([1, 2], name='s1')
    s2 = pd.Series([3, 4], name='s2')
    result = pd.concat([s1, s2], axis=1)
    expected = pd.DataFrame({'s1': [1, 2], 's2': [3, 4]})
    assert_frame_equal(result, expected)

def test_concat_dataframe_axis0():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 5, 6], 'B': [3, 4, 7, 8]})
    assert_frame_equal(result, expected)

def test_concat_dataframe_axis1():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8]})
    assert_frame_equal(result, expected)

def test_concat_ignore_index():
    s1 = pd.Series([1, 2], index=['a', 'b'])
    s2 = pd.Series([3, 4], index=['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3])
    assert_series_equal(result, expected)

def test_concat_keys():
    s1 = pd.Series([1, 2], name='s1')
    s2 = pd.Series([3, 4], name='s2')
    result = pd.concat([s1, s2], keys=['first', 'second'])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples(
        [('first', 0), ('first', 1), ('second', 0), ('second', 1)]))
    assert_series_equal(result, expected)

def test_concat_join_inner():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'A': [1, 2, 5, 6]})
    assert_frame_equal(result, expected)

def test_concat_join_outer():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2], join='outer')
    expected = pd.DataFrame({'A': [1, 2, 5, 6], 'B': [3, 4, np.nan, np.nan], 'C': [np.nan, np.nan, 7, 8]})
    assert_frame_equal(result, expected)

def test_concat_empty_dataframe():
    df1 = pd.DataFrame({'A': []})
    df2 = pd.DataFrame({'B': []})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'A': [], 'B': []})
    assert_frame_equal(result, expected)

def test_concat_mixed_dtypes():
    df1 = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    df2 = pd.DataFrame({'A': [3, 4], 'B': [5.5, 6.6]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['x', 'y', 5.5, 6.6]})
    assert_frame_equal(result, expected)

def test_concat_non_unique_index():
    df1 = pd.DataFrame({'A': [1, 2]}, index=['x', 'x'])
    df2 = pd.DataFrame({'A': [3, 4]}, index=['y', 'y'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=['x', 'x', 'y', 'y'])
    assert_frame_equal(result, expected)

def test_concat_multiindex_rows():
    df1 = pd.DataFrame({'A': [1, 2]}, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)]))
    df2 = pd.DataFrame({'A': [3, 4]}, index=pd.MultiIndex.from_tuples([('y', 1), ('y', 2)]))
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=pd.MultiIndex.from_tuples(
        [('x', 1), ('x', 2), ('y', 1), ('y', 2)]))
    assert_frame_equal(result, expected)

def test_concat_multiindex_columns():
    df1 = pd.DataFrame({'A': [1, 2]}, columns=pd.MultiIndex.from_tuples([('x', 1)]))
    df2 = pd.DataFrame({'B': [3, 4]}, columns=pd.MultiIndex.from_tuples([('y', 1)]))
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, columns=pd.MultiIndex.from_tuples(
        [('x', 1), ('y', 1)]))
    assert_frame_equal(result, expected)

def test_concat_categorical_dtypes():
    df1 = pd.DataFrame({'A': pd.Categorical(['x', 'y'])})
    df2 = pd.DataFrame({'A': pd.Categorical(['y', 'z'])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': pd.Categorical(['x', 'y', 'y', 'z'])})
    assert_frame_equal(result, expected)

def test_concat_datetime_dtypes():
    df1 = pd.DataFrame({'A': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    df2 = pd.DataFrame({'A': pd.to_datetime(['2020-01-03', '2020-01-04'])})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])})
    assert_frame_equal(result, expected)

def test_concat_boolean_mix():
    df1 = pd.DataFrame({'A': [True, False]})
    df2 = pd.DataFrame({'A': [1, 2]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [True, False, 1, 2]})
    assert_frame_equal(result, expected)

def test_concat_verify_integrity():
    df1 = pd.DataFrame({'A': [1]}, index=['x'])
    df2 = pd.DataFrame({'A': [2]}, index=['x'])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_copy_false():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], copy=False)
    # Check that the result is a view (modifying should affect original)
    result.iloc[0, 0] = 99
    assert df1.iloc[0, 0] == 99

def test_concat_sort_false():
    df1 = pd.DataFrame({'A': [1, 2]}, index=['b', 'a'])
    df2 = pd.DataFrame({'A': [3, 4]}, index=['d', 'c'])
    result = pd.concat([df1, df2], sort=False)
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=['b', 'a', 'd', 'c'])
    assert_frame_equal(result, expected)