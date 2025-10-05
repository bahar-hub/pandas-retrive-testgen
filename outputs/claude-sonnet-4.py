import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal


def test_concat_basic_series():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)


def test_concat_basic_dataframes():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 5, 6], 'B': [3, 4, 7, 8]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)


def test_concat_axis_0():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], axis=0)
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)


def test_concat_axis_1():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'B': [3, 4]})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(result, expected)


def test_concat_axis_index():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], axis='index')
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)


def test_concat_axis_columns():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'B': [3, 4]})
    result = pd.concat([df1, df2], axis='columns')
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(result, expected)


def test_concat_ignore_index_true():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series([1, 2, 3, 4])
    assert_series_equal(result, expected)


def test_concat_ignore_index_false():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], ignore_index=False)
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)


def test_concat_join_outer():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2], join='outer')
    expected = pd.DataFrame({
        'A': [1, 2, 5, 6],
        'B': [3.0, 4.0, np.nan, np.nan],
        'C': [np.nan, np.nan, 7.0, 8.0]
    }, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)


def test_concat_join_inner():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'A': [1, 2, 5, 6]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)


def test_concat_keys():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['x', 'y'])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('x', 0), ('x', 1), ('y', 0), ('y', 1)]))
    assert_series_equal(result, expected)


def test_concat_keys_with_names():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['x', 'y'], names=['level1', 'level2'])
    expected = pd.Series([1, 2, 3, 4], 
                        index=pd.MultiIndex.from_tuples([('x', 0), ('x', 1), ('y', 0), ('y', 1)], 
                                                       names=['level1', 'level2']))
    assert_series_equal(result, expected)


def test_concat_verify_integrity_false():
    df1 = pd.DataFrame({'A': [1]}, index=[0])
    df2 = pd.DataFrame({'A': [2]}, index=[0])
    result = pd.concat([df1, df2], verify_integrity=False)
    expected = pd.DataFrame({'A': [1, 2]}, index=[0, 0])
    assert_frame_equal(result, expected)


def test_concat_verify_integrity_true_raises():
    df1 = pd.DataFrame({'A': [1]}, index=[0])
    df2 = pd.DataFrame({'A': [2]}, index=[0])
    with pytest.raises(ValueError, match="Indexes have overlapping values"):
        pd.concat([df1, df2], verify_integrity=True)


def test_concat_sort_false():
    df1 = pd.DataFrame({'B': [1], 'A': [2]})
    df2 = pd.DataFrame({'A': [3], 'B': [4]})
    result = pd.concat([df1, df2], sort=False)
    expected = pd.DataFrame({'B': [1, 4], 'A': [2, 3]}, index=[0, 0])
    assert_frame_equal(result, expected)


def test_concat_sort_true():
    df1 = pd.DataFrame({'B': [1], 'A': [2]})
    df2 = pd.DataFrame({'A': [3], 'B': [4]})
    result = pd.concat([df1, df2], sort=True)
    expected = pd.DataFrame({'A': [2, 3], 'B': [1, 4]}, index=[0, 0])
    assert_frame_equal(result, expected)


def test_concat_empty_list():
    with pytest.raises(ValueError):
        pd.concat([])


def test_concat_single_object():
    df = pd.DataFrame({'A': [1, 2]})
    result = pd.concat([df])
    assert_frame_equal(result, df)


def test_concat_empty_dataframes():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame({'A': [1, 2]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2]})
    assert_frame_equal(result, expected)


def test_concat_empty_series():
    s1 = pd.Series([], dtype='float64')
    s2 = pd.Series([1, 2])
    result = pd.concat([s1, s2])
    expected = pd.Series([1.0, 2.0], index=[0, 1])
    assert_series_equal(result, expected)


def test_concat_mixed_series_dataframe():
    s = pd.Series([1, 2])
    df = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([s, df])
    expected = pd.DataFrame({0: [1.0, 2.0, np.nan, np.nan], 'A': [np.nan, np.nan, 3.0, 4.0]}, 
                           index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)


def test_concat_series_axis_1():
    s1 = pd.Series([1, 2], name='A')
    s2 = pd.Series([3, 4], name='B')
    result = pd.concat([s1, s2], axis=1)
    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert_frame_equal(result, expected)


def test_concat_different_dtypes():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': ['a', 'b']})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 'a', 'b']}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)


def test_concat_multiindex_rows():
    idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2)])
    df1 = pd.DataFrame({'X': [1, 2]}, index=idx)
    df2 = pd.DataFrame({'X': [3, 4]}, index=idx)
    result = pd.concat([df1, df2])
    expected_idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('A', 1), ('A', 2)])
    expected = pd.DataFrame({'X': [1, 2, 3, 4]}, index=expected_idx)
    assert_frame_equal(result, expected)


def test_concat_multiindex_columns():
    cols = pd.MultiIndex.from_tuples([('A', 'X'), ('A', 'Y')])
    df1 = pd.DataFrame([[1, 2]], columns=cols)
    df2 = pd.DataFrame([[3, 4]], columns=cols)
    result = pd.concat([df1, df2])
    expected = pd.DataFrame([[1, 2], [3, 4]], columns=cols, index=[0, 0])
    assert_frame_equal(result, expected)


def test_concat_categorical_dtype():
    cat1 = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    cat2 = pd.Categorical(['c', 'a'], categories=['a', 'b', 'c'])
    s1 = pd.Series(cat1)
    s2 = pd.Series(cat2)
    result = pd.concat([s1, s2])
    expected_cat = pd.Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'])
    expected = pd.Series(expected_cat, index=[0, 1, 0, 1])
    assert_series_equal(result, expected)


def test_concat_datetime_dtype():
    dates1 = pd.to_datetime(['2020-01-01', '2020-01-02'])
    dates2 = pd.to_datetime(['2020-01-03', '2020-01-04'])
    s1 = pd.Series(dates1)
    s2 = pd.Series(dates2)
    result = pd.concat([s1, s2])
    expected = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']), 
                        index=[0, 1, 0, 1])
    assert_series_equal(result, expected)


def test_concat_timezone_aware():
    tz_dates1 = pd.to_datetime(['2020-01-01', '2020-01-02']).tz_localize('UTC')
    tz_dates2 = pd.to_datetime(['2020-01-03', '2020-01-04']).tz_localize('UTC')
    s1 = pd.Series(tz_dates1)
    s2 = pd.Series(tz_dates2)
    result = pd.concat([s1, s2])
    expected_dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']).tz_localize('UTC')
    expected = pd.Series(expected_dates, index=[0, 1, 0, 1])
    assert_series_equal(result, expected)


def test_concat_boolean_dtype():
    s1 = pd.Series([True, False])
    s2 = pd.Series([False, True])
    result = pd.concat([s1, s2])
    expected = pd.Series([True, False, False, True], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)


def test_concat_object_dtype():
    s1 = pd.Series(['a', 'b'], dtype='object')
    s2 = pd.Series(['c', 'd'], dtype='object')
    result = pd.concat([s1, s2])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=[0, 1, 0, 1], dtype='object')
    assert_series_equal(result, expected)


def test_concat_numeric_mixed():
    s1 = pd.Series([1, 2], dtype='int64')
    s2 = pd.Series([3.5, 4.5], dtype='float64')
    result = pd.concat([s1, s2])
    expected = pd.Series([1.0, 2.0, 3.5, 4.5], index=[0, 1, 0, 1], dtype='float64')
    assert_series_equal(result, expected)


def test_concat_non_unique_indices():
    df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 0])
    df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 1])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({'A': [1, 2, 3, 4]}, index=[0, 0, 1, 1])
    assert_frame_equal(result, expected)


def test_concat_non_unique_columns():
    df1 = pd.DataFrame([[1, 2]], columns=['A', 'A'])
    df2 = pd.DataFrame([[3, 4]], columns=['A', 'A'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'A'], index=[0, 0])
    assert_frame_equal(result, expected)


def test_concat_differing_columns():
    df1 = pd.DataFrame({'A': [1], 'B': [2]})
    df2 = pd.DataFrame({'B': [3], 'C': [4]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({
        'A': [1.0, np.nan],
        'B': [2, 3],
        'C': [np.nan, 4.0]
    }, index=[0, 0])
    assert_frame_equal(result, expected)


def test_concat_mapping_input():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat({'x': s1, 'y': s2})
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('x', 0), ('x', 1), ('y', 0), ('y', 1)]))
    assert_series_equal(result, expected)


def test_concat_mapping_with_keys():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat({'x': s1, 'y': s2}, keys=['a', 'b'])
    expected = pd.Series([1, 2, 3, 4], index=pd.MultiIndex.from_tuples([('a', 0), ('a', 1), ('b', 0), ('b', 1)]))
    assert_series_equal(result, expected)


def test_concat_levels_parameter():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['x', 'y'], levels=[['x', 'y', 'z']])
    expected_idx = pd.MultiIndex.from_tuples([('x', 0), ('x', 1), ('y', 0), ('y', 1)], 
                                           levels=[['x', 'y', 'z'], [0, 1]])
    expected = pd.Series([1, 2, 3, 4], index=expected_idx)
    assert_series_equal(result, expected)


def test_concat_none_objects_dropped():
    s1 = pd.Series([1, 2])
    s2 = None
    s3 = pd.Series([3, 4])
    result = pd.concat([s1, s2, s3])
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)


def test_concat_all_none_raises():
    with pytest.raises(ValueError):
        pd.concat([None, None])


def test_concat_copy_false():
    df = pd.DataFrame({'A': [1, 2]})
    result = pd.concat([df], copy=False)
    assert_frame_equal(result, df)


def test_concat_copy_true():
    df = pd.DataFrame({'A': [1, 2]})
    result = pd.concat([df], copy=True)
    assert_frame_equal(result, df)


def test_concat_keys_tuples():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=[('A', 1), ('A', 2)])
    expected_idx = pd.MultiIndex.from_tuples([(('A', 1), 0), (('A', 1), 1), (('A', 2), 0), (('A', 2), 1)])
    expected = pd.Series([1, 2, 3, 4], index=expected_idx)
    assert_series_equal(result, expected)


def test_concat_datetime_index_sort():
    dates1 = pd.date_range('2020-01-01', periods=2)
    dates2 = pd.date_range('2019-12-30', periods=2)
    df1 = pd.DataFrame({'A': [1, 2]}, index=dates1)
    df2 = pd.DataFrame({'B': [3, 4]}, index=dates2)
    result = pd.concat([df1, df2], axis=1, join='outer')
    expected_idx = pd.DatetimeIndex(['2019-12-30', '2019-12-31', '2020-01-01', '2020-01-02'])
    expected = pd.DataFrame({
        'A': [np.nan, np.nan, 1.0, 2.0],
        'B': [3.0, 4.0, np.nan, np.nan]
    }, index=expected_idx)
    assert_frame_equal(result, expected)


def test_concat_return_series_when_all_series():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2])
    assert isinstance(result, pd.Series)


def test_concat_return_dataframe_when_mixed():
    s = pd.Series([1, 2])
    df = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([s, df])
    assert isinstance(result, pd.DataFrame)


def test_concat_return_dataframe_when_axis_1():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], axis=1)
    assert isinstance(result, pd.DataFrame)


def test_concat_names_single_level():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['x', 'y'], names=['key'])
    expected_idx = pd.MultiIndex.from_tuples([('x', 0), ('x', 1), ('y', 0), ('y', 1)], names=['key', None])
    expected = pd.Series([1, 2, 3, 4], index=expected_idx)
    assert_series_equal(result, expected)


def test_concat_ignore_index_with_keys():
    s1 = pd.Series([1, 2])
    s2 = pd.Series([3, 4])
    result = pd.concat([s1, s2], keys=['x', 'y'], ignore_index=True)
    expected = pd.Series([1, 2, 3, 4])
    assert_series_equal(result, expected)


def test_concat_different_index_types():
    s1 = pd.Series([1, 2], index=[0, 1])
    s2 = pd.Series([3, 4], index=['a', 'b'])
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 'a', 'b'])
    assert_series_equal(result, expected)