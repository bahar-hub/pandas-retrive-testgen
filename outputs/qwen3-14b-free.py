import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

def test_concat_axis_0():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]}, index=[2, 3])
    result = pd.concat([df1, df2], axis=0)
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]}, index=[0, 1, 2, 3])
    assert_frame_equal(result, expected)

def test_concat_axis_1():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'c': [5, 6], 'd': [7, 8]}, index=[0, 1])
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]}, index=[0, 1])
    assert_frame_equal(result, expected)

def test_concat_series_axis_0():
    s1 = pd.Series([1, 2], index=[0, 1])
    s2 = pd.Series([3, 4], index=[2, 3])
    result = pd.concat([s1, s2], axis=0)
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3])
    assert_series_equal(result, expected)

def test_concat_series_and_df():
    s = pd.Series([1, 2], index=[0, 1])
    df = pd.DataFrame({'a': [3, 4]}, index=[2, 3])
    result = pd.concat([s, df], axis=0)
    expected = pd.DataFrame({'0': [1, 2, np.nan, np.nan], 'a': [np.nan, np.nan, 3, 4]}, index=[0, 1, 2, 3])
    assert_frame_equal(result, expected)

def test_concat_keys_single_level():
    s1 = pd.Series([1, 2], index=[0, 1], name='s1')
    s2 = pd.Series([3, 4], index=[2, 3], name='s2')
    result = pd.concat([s1, s2], keys=['a', 'b'])
    expected = pd.DataFrame({'s1': [1, 2], 's2': [3, 4]}, index=[[0, 1, 2, 3], ['a', 'a', 'b', 'b']])
    assert_frame_equal(result, expected)

def test_concat_keys_with_names():
    s1 = pd.Series([1, 2], index=[0, 1], name='s1')
    s2 = pd.Series([3, 4], index=[2, 3], name='s2')
    result = pd.concat([s1, s2], keys=['a', 'b'], names=['Group', 'Original Index'])
    expected = pd.DataFrame({'s1': [1, 2], 's2': [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 'a'), (1, 'a'), (2, 'b'), (3, 'b')], names=['Original Index', 'Group']))
    assert_frame_equal(result, expected)

def test_concat_ignore_index():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 0])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[0, 0])
    result = pd.concat([df1, df2], ignore_index=True)
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=[0, 1, 2, 3])
    assert_frame_equal(result, expected)

def test_concat_inner_join():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'b': [5, 6], 'c': [7, 8]}, index=[0, 1])
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'b': [3, 4, 5, 6]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_outer_join():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'c': [5, 6], 'd': [7, 8]}, index=[2, 3])
    result = pd.concat([df1, df2], join='outer')
    expected = pd.DataFrame({'a': [1, 2, np.nan, np.nan], 'b': [3, 4, np.nan, np.nan], 'c': [np.nan, np.nan, 5, 6], 'd': [np.nan, np.nan, 7, 8]}, index=[0, 1, 2, 3])
    assert_frame_equal(result, expected)

def test_concat_sort_non_aligned():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[2, 1])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[0, 3])
    result = pd.concat([df1, df2], axis=0, sort=True)
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=[0, 1, 2, 3])
    assert_frame_equal(result, expected)

def test_concat_sort_datetime_outer():
    df1 = pd.DataFrame({'a': [1, 2]}, index=pd.to_datetime(['2020-01-02', '2020-01-01']))
    df2 = pd.DataFrame({'a': [3, 4]}, index=pd.to_datetime(['2020-01-03', '2020-01-04']))
    result = pd.concat([df1, df2], axis=0, join='outer', sort=False)
    expected = pd.DataFrame({'a': [1, 2, 3, 4]}, index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']))
    assert_frame_equal(result, expected)

def test_concat_verify_integrity_duplicate_index():
    df1 = pd.DataFrame({'a': [1, 2]}, index=['a', 'a'])
    df2 = pd.DataFrame({'a': [3, 4]}, index=['a', 'a'])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_empty_df():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    result = pd.concat([df1, df2])
    assert_frame_equal(result, pd.DataFrame())

def test_concat_empty_series():
    s1 = pd.Series([])
    s2 = pd.Series([])
    result = pd.concat([s1, s2])
    assert_series_equal(result, pd.Series([]))

def test_concat_mixed_dtypes():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'a': [5, 6], 'b': ['x', 'y']})
    result = pd.concat([df1, df2])
    assert_frame_equal(result, pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 'x', 'y']}, index=[0, 1, 0, 1]))
    assert result.dtypes['b'] == object

def test_concat_non_unique_indices():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 0])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[0, 0])
    result = pd.concat([df1, df2])
    assert result.index.has_duplicates

def test_concat_multiindex_rows():
    index1 = pd.MultiIndex.from_product([[1, 2], ['a', 'b']], names=['level1', 'level2'])
    df1 = pd.DataFrame({'x': [10, 20, 30, 40]}, index=index1)
    index2 = pd.MultiIndex.from_product([[3, 4], ['c', 'd']], names=['level1', 'level2'])
    df2 = pd.DataFrame({'x': [50, 60, 70, 80]}, index=index2)
    result = pd.concat([df1, df2])
    expected_index = index1.append(index2)
    assert result.index.equals(expected_index)

def test_concat_multiindex_columns():
    columns1 = pd.MultiIndex.from_product([['A', 'B'], ['x', 'y']])
    df1 = pd.DataFrame([[1, 2, 3, 4]], columns=columns1)
    columns2 = pd.MultiIndex.from_product([['C', 'D'], ['x', 'y']])
    df2 = pd.DataFrame([[5, 6, 7, 8]], columns=columns2)
    result = pd.concat([df1, df2], axis=1)
    expected_columns = columns1.append(columns2)
    assert result.columns.equals(expected_columns)

def test_concat_categorical_dtype():
    s1 = pd.Series(['a', 'b', 'c'], dtype='category')
    s2 = pd.Series(['d', 'e', 'f'], dtype='category')
    result = pd.concat([s1, s2])
    assert result.dtype == 'category'

def test_concat_datetime_timezone():
    s1 = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02'], tz='UTC'))
    s2 = pd.Series(pd.to_datetime(['2020-01-03', '2020-01-04'], tz='UTC'))
    result = pd.concat([s1, s2])
    assert result.dtype == 'datetime64[ns, UTC]'

def test_concat_boolean_object_mix():
    df1 = pd.DataFrame({'a': [True, False], 'b': [1, 2]})
    df2 = pd.DataFrame({'a': ['x', 'y'], 'b': [3, 4]})
    result = pd.concat([df1, df2])
    assert result.dtypes['a'] == object

def test_concat_all_none():
    with pytest.raises(ValueError):
        pd.concat([None, None])

def test_concat_with_none_objects():
    df1 = pd.DataFrame({'a': [1]})
    result = pd.concat([df1, None])
    assert_frame_equal(result, df1)

def test_concat_mapping_objs():
    df1 = pd.DataFrame({'a': [1]}, index=[0])
    df2 = pd.DataFrame({'b': [2]}, index=[1])
    result = pd.concat({'df1': df1, 'df2': df2}, axis=0)
    expected = pd.concat([df1, df2], keys=['df1', 'df2'])
    assert_frame_equal(result, expected)

def test_concat_mapping_keys_sorted():
    df1 = pd.DataFrame({'a': [1]}, index=[0])
    df2 = pd.DataFrame({'b': [2]}, index=[1])
    result = pd.concat({'df2': df2, 'df1': df1}, axis=0)
    expected = pd.concat([df1, df2], keys=['df1', 'df2'])
    assert_frame_equal(result, expected)

def test_concat_levels():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[2, 3])
    result = pd.concat([df1, df2], keys=[1, 2], levels=[0, 1])
    expected_index = pd.MultiIndex.from_product([[1, 2], [0, 1, 2, 3]], levels=[0, 1])
    assert result.index.equals(expected_index)

def test_concat_names():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[2, 3])
    result = pd.concat([df1, df2], keys=[1, 2], names=['Group', 'Index'])
    expected_index = pd.MultiIndex.from_tuples([(0, 1), (1, 1), (2, 2), (3, 2)], names=['Index', 'Group'])
    assert result.index.equals(expected_index)

def test_concat_differing_columns():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'b': [5, 6], 'c': [7, 8]}, index=[0, 1])
    result = pd.concat([df1, df2], join='outer')
    expected = pd.DataFrame({'a': [1, 2, np.nan, np.nan], 'b': [3, 4, 5, 6], 'c': [np.nan, np.nan, 7, 8]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_inner_join_differing_columns():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'b': [5, 6], 'c': [7, 8]}, index=[0, 1])
    result = pd.concat([df1, df2], join='inner')
    expected = pd.DataFrame({'b': [3, 4, 5, 6]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_non_unique_columns():
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]}, index=[0, 1])
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({'a': [1, 2, 5, 6], 'b': [3, 4, 7, 8]}, index=[0, 1])
    assert_frame_equal(result, expected)

def test_concat_with_copy():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[2, 3])
    result = pd.concat([df1, df2], copy=False)
    assert not result._mgr._data.is_copy

def test_concat_with_verify_integrity():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({'a': [3, 4]}, index=[0, 1])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_series_with_keys():
    s1 = pd.Series([1, 2], index=[0, 1], name='s1')
    s2 = pd.Series([3, 4], index=[2, 3], name='s2')
    result = pd.concat([s1, s2], keys=['a', 'b'])
    expected = pd.DataFrame({'s1': [1, 2], 's2': [3, 4]}, index=[[0, 1, 2, 3], ['a', 'a', 'b', 'b']])
    assert_frame_equal(result, expected)

def test_concat_series_with_keys_and_names():
    s1 = pd.Series([1, 2], index=[0, 1], name='s1')
    s2 = pd.Series([3, 4], index=[2, 3], name='s2')
    result = pd.concat([s1, s2], keys=['a', 'b'], names=['Group', 'Original Index'])
    expected = pd.DataFrame({'s1': [1, 2], 's2': [3, 4]}, index=pd.MultiIndex.from_tuples([(0, 'a'), (1, 'a'), (2, 'b'), (3, 'b')], names=['Original Index', 'Group']))
    assert_frame_equal(result, expected)