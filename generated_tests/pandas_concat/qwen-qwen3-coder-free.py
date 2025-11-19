import pytest
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from typing import Any, Dict, List, Union

def test_concat_basic_series() -> None:
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = Series(['a', 'b', 'c', 'd'], index=[0, 1, 0, 1])
    assert_series_equal(result, expected)

def test_concat_ignore_index_series() -> None:
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = Series(['a', 'b', 'c', 'd'], index=[0, 1, 2, 3])
    assert_series_equal(result, expected)

def test_concat_keys_series() -> None:
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    expected_index = pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_names_series() -> None:
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series name', 'Row ID'])
    expected_index = pd.MultiIndex.from_tuples(
        [('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)],
        names=['Series name', 'Row ID']
    )
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_basic_dataframe() -> None:
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
    result = pd.concat([df1, df2])
    expected = DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], columns=['letter', 'number'])
    expected.index = [0, 1, 0, 1]
    assert_frame_equal(result, expected)

def test_concat_ignore_index_dataframe() -> None:
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
    result = pd.concat([df1, df2], ignore_index=True)
    expected = DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], columns=['letter', 'number'])
    expected.index = [0, 1, 2, 3]
    assert_frame_equal(result, expected)

def test_concat_axis1_dataframe() -> None:
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = DataFrame([['bird', 'polly'], ['monkey', 'george']], columns=['animal', 'name'])
    result = pd.concat([df1, df2], axis=1)
    expected = DataFrame([['a', 1, 'bird', 'polly'], ['b', 2, 'monkey', 'george']], 
                        columns=['letter', 'number', 'animal', 'name'])
    assert_frame_equal(result, expected)

def test_concat_join_inner() -> None:
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df3 = DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
    result = pd.concat([df1, df3], join="inner")
    expected = DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], columns=['letter', 'number'])
    expected.index = [0, 1, 0, 1]
    assert_frame_equal(result, expected)

def test_concat_join_outer() -> None:
    df1 = DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df3 = DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
    result = pd.concat([df1, df3], join="outer")
    expected = DataFrame([['a', 1, np.nan], ['b', 2, np.nan], ['c', 3, 'cat'], ['d', 4, 'dog']], 
                        columns=['letter', 'number', 'animal'])
    expected.index = [0, 1, 0, 1]
    assert_frame_equal(result, expected)

def test_concat_sort_false() -> None:
    df1 = DataFrame([[1, 2]], columns=['b', 'a'])
    df2 = DataFrame([[3, 4]], columns=['a', 'b'])
    result = pd.concat([df1, df2], sort=False)
    expected = DataFrame([[1, 2], [3, 4]], columns=['b', 'a'])
    expected.index = [0, 0]
    assert_frame_equal(result, expected)

def test_concat_sort_true() -> None:
    df1 = DataFrame([[1, 2]], columns=['b', 'a'])
    df2 = DataFrame([[3, 4]], columns=['a', 'b'])
    result = pd.concat([df1, df2], sort=True)
    expected = DataFrame([[2, 1], [4, 3]], columns=['a', 'b'])
    expected.index = [0, 0]
    assert_frame_equal(result, expected)

def test_concat_verify_integrity() -> None:
    df5 = DataFrame([1], index=['a'])
    df6 = DataFrame([2], index=['a'])
    with pytest.raises(ValueError, match="Indexes have overlapping values"):
        pd.concat([df5, df6], verify_integrity=True)

def test_concat_empty_objects() -> None:
    s1 = Series([], dtype=object)
    s2 = Series(['a', 'b'])
    result = pd.concat([s1, s2])
    expected = Series(['a', 'b'], index=[0, 1])
    assert_series_equal(result, expected)

def test_concat_all_empty() -> None:
    s1 = Series([], dtype=object)
    s2 = Series([], dtype=object)
    result = pd.concat([s1, s2])
    expected = Series([], dtype=object)
    assert_series_equal(result, expected)

def test_concat_mixed_dtypes() -> None:
    s1 = Series([1, 2], dtype=int)
    s2 = Series(['a', 'b'], dtype=object)
    result = pd.concat([s1, s2])
    expected = Series([1, 2, 'a', 'b'], index=[0, 1, 0, 1], dtype=object)
    assert_series_equal(result, expected)

def test_concat_non_unique_index() -> None:
    s1 = Series(['a', 'b'], index=[0, 0])
    s2 = Series(['c', 'd'], index=[1, 1])
    result = pd.concat([s1, s2])
    expected = Series(['a', 'b', 'c', 'd'], index=[0, 0, 1, 1])
    assert_series_equal(result, expected)

def test_concat_multiindex_rows() -> None:
    index1 = pd.MultiIndex.from_tuples([('A', 1), ('A', 2)])
    df1 = DataFrame([[1, 2], [3, 4]], index=index1, columns=['a', 'b'])
    index2 = pd.MultiIndex.from_tuples([('B', 1), ('B', 2)])
    df2 = DataFrame([[5, 6], [7, 8]], index=index2, columns=['a', 'b'])
    result = pd.concat([df1, df2])
    expected_index = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2)])
    expected = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=expected_index, columns=['a', 'b'])
    assert_frame_equal(result, expected)

def test_concat_multiindex_columns() -> None:
    columns1 = pd.MultiIndex.from_tuples([('X', 'a'), ('X', 'b')])
    df1 = DataFrame([[1, 2], [3, 4]], columns=columns1)
    columns2 = pd.MultiIndex.from_tuples([('Y', 'a'), ('Y', 'b')])
    df2 = DataFrame([[5, 6], [7, 8]], columns=columns2)
    result = pd.concat([df1, df2], axis=0)
    expected_columns = pd.MultiIndex.from_tuples([('X', 'a'), ('X', 'b'), ('Y', 'a'), ('Y', 'b')])
    expected_data = [[1, 2, np.nan, np.nan], [3, 4, np.nan, np.nan], 
                     [np.nan, np.nan, 5, 6], [np.nan, np.nan, 7, 8]]
    expected = DataFrame(expected_data, columns=expected_columns, index=[0, 1, 0, 1])
    assert_frame_equal(result, expected)

def test_concat_keys_with_multiindex() -> None:
    df1 = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    df2 = DataFrame([[5, 6], [7, 8]], columns=['a', 'b'])
    result = pd.concat([df1, df2], keys=['first', 'second'])
    expected_index = pd.MultiIndex.from_tuples([('first', 0), ('first', 1), ('second', 0), ('second', 1)])
    expected = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], columns=['a', 'b'], index=expected_index)
    assert_frame_equal(result, expected)

def test_concat_levels() -> None:
    df1 = DataFrame([[1, 2]], columns=['a', 'b'])
    df2 = DataFrame([[3, 4]], columns=['a', 'b'])
    result = pd.concat([df1, df2], keys=[('X', '1'), ('Y', '2')], levels=[['X', 'Y', 'Z'], ['1', '2', '3']])
    assert result.index.levels[0].tolist() == ['X', 'Y', 'Z']
    assert result.index.levels[1].tolist() == ['1', '2', '3']

def test_concat_mapping_input() -> None:
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat({'first': s1, 'second': s2})
    expected_index = pd.MultiIndex.from_tuples([('first', 0), ('first', 1), ('second', 0), ('second', 1)])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_mapping_with_keys_override() -> None:
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat({'first': s1, 'second': s2}, keys=['A', 'B'])
    expected_index = pd.MultiIndex.from_tuples([('A', 0), ('A', 1), ('B', 0), ('B', 1)])
    expected = Series(['a', 'b', 'c', 'd'], index=expected_index)
    assert_series_equal(result, expected)

def test_concat_categorical_dtypes() -> None:
    s1 = Series(pd.Categorical(['a', 'b'], categories=['a', 'b', 'c']))
    s2 = Series(pd.Categorical(['c', 'a'], categories=['a', 'b', 'c']))
    result = pd.concat([s1, s2])
    expected = Series(pd.Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c']), 
                      index=[0, 1, 0, 1])
    assert_series_equal(result, expected)

def test_concat_datetime_dtypes() -> None:
    s1 = Series(pd.to_datetime(['2020-01-01', '2020-01-02']))
    s2 = Series(pd.to_datetime(['2020-01-03', '2020-01-04']))
    result = pd.concat([s1, s2])
    expected = Series(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']), 
                      index=[0, 1, 0, 1])
    assert_series_equal(result, expected)

def test_concat_none_objects() -> None:
    s1 = Series(['a', 'b'])
    with pytest.raises(ValueError, match="No objects to concatenate"):
        pd.concat([None, None])

def test_concat_single_object() -> None:
    s1 = Series(['a', 'b'])
    result = pd.concat([s1])
    assert_series_equal(result, s1)

def test_concat_copy_behavior() -> None:
    df1 = DataFrame([[1, 2]], columns=['a', 'b'])
    df2 = DataFrame([[3, 4]], columns=['a', 'b'])
    result = pd.concat([df1, df2], copy=False)
    expected = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    expected.index = [0, 0]
    assert_frame_equal(result, expected)