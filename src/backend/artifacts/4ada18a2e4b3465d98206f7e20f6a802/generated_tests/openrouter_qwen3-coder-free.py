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
    expected = DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], columns=['letter', 'number'], index=[0, 1, 0, 1])
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

def test_concat_single_object():
    df = DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = pd.concat([df])
    assert_frame_equal(result, df)

def test_concat_sort_true():
    df1 = DataFrame({'b': [1], 'a': [2]})
    df2 = DataFrame({'a': [3], 'c': [4]})
    result = pd.concat([df1, df2], sort=True)
    expected = DataFrame({'a': [2, 3], 'b': [1, None], 'c': [None, 4]}, index=[0, 0])
    assert_frame_equal(result, expected)

def test_concat_sort_false():
    df1 = DataFrame({'b': [1], 'a': [2]})
    df2 = DataFrame({'a': [3], 'c': [4]})
    result = pd.concat([df1, df2], sort=False)
    expected = DataFrame({'b': [1, None], 'a': [2, 3], 'c': [None, 4]}, index=[0, 0])
    assert_frame_equal(result, expected)

def test_concat_with_mapping():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat({'first': s1, 'second': s2})
    expected_index = pd.MultiIndex.from_tuples([('first', 0), ('first', 1), ('second', 0), ('second', 1)])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_with_mapping_and_keys():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat({'first': s1, 'second': s2}, keys=['second', 'first'])
    expected_index = pd.MultiIndex.from_tuples([('second', 0), ('second', 1), ('first', 0), ('first', 1)])
    expected = Series(['c', 'd', 'a', 'b'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_mixed_dtypes():
    df1 = DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    df2 = DataFrame({'A': [3.0, 4.0], 'B': [True, False]})
    result = pd.concat([df1, df2])
    expected = DataFrame({'A': [1, 2, 3.0, 4.0], 'B': ['x', 'y', True, False]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_empty_dataframe():
    df1 = DataFrame(columns=['A', 'B'])
    df2 = DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    result = pd.concat([df1, df2])
    expected = DataFrame({'A': [1, 2], 'B': ['x', 'y']}, index=[0, 1])
    assert_frame_equal(result, expected)

def test_concat_multiindex_columns():
    df1 = DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([('A', 'a'), ('B', 'b')]))
    df2 = DataFrame([[3, 4]], columns=pd.MultiIndex.from_tuples([('A', 'a'), ('B', 'b')]))
    result = pd.concat([df1, df2])
    expected = DataFrame([[1, 2], [3, 4]], columns=pd.MultiIndex.from_tuples([('A', 'a'), ('B', 'b')]), index=[0, 0])
    assert_frame_equal(result, expected)

def test_concat_multiindex_index():
    index1 = pd.MultiIndex.from_tuples([('x', 1), ('x', 2)])
    index2 = pd.MultiIndex.from_tuples([('y', 1), ('y', 2)])
    df1 = DataFrame({'A': [1, 2]}, index=index1)
    df2 = DataFrame({'A': [3, 4]}, index=index2)
    result = pd.concat([df1, df2])
    expected_index = pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)])
    expected = DataFrame({'A': [1, 2, 3, 4]}, index=expected_index)
    assert_frame_equal(result, expected)

def test_concat_datetime_index():
    df1 = DataFrame({'A': [1, 2]}, index=pd.to_datetime(['2020-01-01', '2020-01-02']))
    df2 = DataFrame({'A': [3, 4]}, index=pd.to_datetime(['2020-01-03', '2020-01-04']))
    result = pd.concat([df1, df2])
    expected_index = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
    expected = DataFrame({'A': [1, 2, 3, 4]}, index=expected_index)
    assert_frame_equal(result, expected)

def test_concat_categorical_data():
    df1 = DataFrame({'A': pd.Categorical(['a', 'b'])})
    df2 = DataFrame({'A': pd.Categorical(['c', 'd'])})
    result = pd.concat([df1, df2])
    expected = DataFrame({'A': pd.Categorical(['a', 'b', 'c', 'd'])})
    assert_frame_equal(result, expected)

def test_concat_with_levels():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=[('X', 1), ('Y', 2)], levels=[['X', 'Y', 'Z'], [1, 2, 3]], names=['Level1', 'Level2'])
    expected_index = pd.MultiIndex.from_tuples([('X', 1, 0), ('X', 1, 1), ('Y', 2, 0), ('Y', 2, 1)], names=['Level1', 'Level2', None])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_copy_false():
    df1 = DataFrame({'A': [1, 2]})
    df2 = DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], copy=False)
    expected = DataFrame({'A': [1, 2, 3, 4]}, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)