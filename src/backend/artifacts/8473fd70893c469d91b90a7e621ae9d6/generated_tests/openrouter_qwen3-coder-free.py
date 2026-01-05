import pytest
import pandas as pd
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from typing import Iterable, Mapping, Hashable

def test_concat_basic_series():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = Series(['a', 'b', 'c', 'd'], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)

def test_concat_ignore_index_series():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = Series(['a', 'b', 'c', 'd'], index=[0, 1, 2, 3])
    assert_series_equal(result, expected)

def test_concat_keys_series():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    expected = Series(['a', 'b', 'c', 'd'], index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)]))
    assert_series_equal(result, expected)

def test_concat_names_series():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series name', 'Row ID'])
    expected_index = pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)], names=['Series name', 'Row ID'])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_basic_dataframe():
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
    result = pd.concat([df1, df2])
    expected = DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], columns=['letter', 'number'], index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_join_inner():
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df3 = DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
    result = pd.concat([df1, df3], join="inner")
    expected = DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], columns=['letter', 'number'])
    assert_frame_equal(result, expected)

def test_concat_axis_1():
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df4 = DataFrame([['bird', 'polly'], ['monkey', 'george']], columns=['animal', 'name'])
    result = pd.concat([df1, df4], axis=1)
    expected = DataFrame([['a', 1, 'bird', 'polly'], ['b', 2, 'monkey', 'george']], columns=['letter', 'number', 'animal', 'name'])
    assert_frame_equal(result, expected)

def test_concat_verify_integrity():
    df5 = DataFrame([1], index=['a'])
    df6 = DataFrame([2], index=['a'])
    with pytest.raises(ValueError, match="Indexes have overlapping values"):
        pd.concat([df5, df6], verify_integrity=True)

def test_concat_empty_list():
    with pytest.raises(ValueError, match="No objects to concatenate"):
        pd.concat([])

def test_concat_all_none():
    with pytest.raises(ValueError, match="All objects passed were None"):
        pd.concat([None, None])

def test_concat_with_none():
    s1 = Series(['a', 'b'])
    result = pd.concat([s1, None])
    expected = Series(['a', 'b'], index=[0, 1])
    assert_series_equal(result, expected)

def test_concat_sort_false():
    df1 = DataFrame({'b': [1], 'a': [2]})
    df2 = DataFrame({'a': [3], 'b': [4]})
    result = pd.concat([df1, df2], sort=False)
    expected = DataFrame({'b': [1, 4], 'a': [2, 3]}, index=[0, 0])
    assert_frame_equal(result, expected)

def test_concat_sort_true():
    df1 = DataFrame({'b': [1], 'a': [2]})
    df2 = DataFrame({'a': [3], 'b': [4]})
    result = pd.concat([df1, df2], sort=True)
    expected = DataFrame({'a': [2, 3], 'b': [1, 4]}, index=[0, 0])
    assert_frame_equal(result, expected)

def test_concat_mixed_dtypes():
    s1 = Series([1, 2], dtype='int64')
    s2 = Series([3.0, 4.0], dtype='float64')
    result = pd.concat([s1, s2])
    expected = Series([1, 2, 3.0, 4.0], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)

def test_concat_datetime_index():
    dates1 = pd.date_range('2020-01-01', periods=2)
    dates2 = pd.date_range('2020-01-03', periods=2)
    s1 = Series([1, 2], index=dates1)
    s2 = Series([3, 4], index=dates2)
    result = pd.concat([s1, s2])
    expected = Series([1, 2, 3, 4], index=pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']))
    assert_series_equal(result, expected)

def test_concat_categorical():
    s1 = Series(pd.Categorical(['a', 'b']))
    s2 = Series(pd.Categorical(['c', 'd']))
    result = pd.concat([s1, s2])
    expected = Series(pd.Categorical(['a', 'b', 'c', 'd']), index=[0, 1, 0, 1])
    assert_series_equal(result, expected)

def test_concat_mapping():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    mapping = {'first': s1, 'second': s2}
    result = pd.concat(mapping)
    expected_index = pd.MultiIndex.from_tuples([('first', 0), ('first', 1), ('second', 0), ('second', 1)])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_mapping_with_keys():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    mapping = {'first': s1, 'second': s2}
    result = pd.concat(mapping, keys=['x', 'y'])
    expected_index = pd.MultiIndex.from_tuples([('x', 0), ('x', 1), ('y', 0), ('y', 1)])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_single_object():
    s1 = Series(['a', 'b'])
    result = pd.concat([s1])
    expected = Series(['a', 'b'], index=[0, 1])
    assert_series_equal(result, expected)

def test_concat_levels():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=[('A', 1), ('A', 2)], levels=[['A'], [1, 2, 3]])
    expected_index = pd.MultiIndex.from_tuples([('A', 1, 0), ('A', 1, 1), ('A', 2, 0), ('A', 2, 1)], levels=[['A'], [1, 2, 3]])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)