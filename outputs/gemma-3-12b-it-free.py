import pytest
import pandas as pd
import numpy as np

def test_concat_series_basic():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    assert_series_equal(result, pd.Series(['a', 'b', 'c', 'd']))

def test_concat_series_ignore_index():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    assert_series_equal(result, pd.Series([0, 1, 2, 3], dtype=int))

def test_concat_series_keys():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    expected = pd.Series(
        ['a', 'b', 'c', 'd'],
        index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)])
    )
    assert_series_equal(result, expected)

def test_concat_series_names():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series name', 'Row ID'])
    expected = pd.Series(
        ['a', 'b', 'c', 'd'],
        index=pd.MultiIndex.from_tuples([('s1', 0), ('s1', 1), ('s2', 0), ('s2', 1)],
                                        names=['Series name', 'Row ID'])
    )
    assert_series_equal(result, expected)

def test_concat_dataframe_basic():
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
    result = pd.concat([df1, df2])
    expected = pd.DataFrame([['a', 1], ['b', 2], ['c', 3], ['d', 4]], columns=['letter', 'number'])
    assert_frame_equal(result, expected)

def test_concat_dataframe_inner_join():
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
    result = pd.concat([df1, df2], join="inner")
    expected = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    assert_frame_equal(result, expected)

def test_concat_dataframe_sort_false():
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
    result = pd.concat([df1, df2], sort=False)
    expected = pd.DataFrame([['a', 1, np.nan], ['b', 2, np.nan], ['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
    assert_frame_equal(result, expected)

def test_concat_dataframe_axis_1():
    df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
    df2 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']], columns=['animal', 'name'])
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame([['a', 1, 'bird', 'polly'], ['b', 2, 'monkey', 'george']], columns=['letter', 'number', 'animal', 'name'])
    assert_frame_equal(result, expected)

def test_concat_verify_integrity():
    df1 = pd.DataFrame([1], index=['a'])
    df2 = pd.DataFrame([2], index=['a'])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_concat_empty_objects():
    s1 = pd.Series([])
    s2 = pd.Series([])
    result = pd.concat([s1, s2])
    assert_series_equal(result, pd.Series([]))

def test_concat_mixed_dtypes():
    df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    df2 = pd.DataFrame({'a': [3, 4], 'c': [True, False]})
    result = pd.concat([df1, df2], sort=False)
    expected = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['x', 'y', np.nan, np.nan], 'c': [np.nan, np.nan, True, False]})
    assert_frame_equal(result, expected)

def test_concat_non_unique_indices():
    df1 = pd.DataFrame({'a': [1, 2]}, index=[0, 0])
    df2 = pd.DataFrame({'b': [3, 4]}, index=[0, 0])
    result = pd.concat([df1, df2], join='outer')
    expected = pd.DataFrame({'a': [1.0, 2.0, np.nan, np.nan], 'b': [np.nan, np.nan, 3.0, 4.0]}, index=[0, 0])
    assert_frame_equal(result, expected)

def test_concat_categorical_dtype():
    cat = pd.Categorical(['a', 'b', 'a'])
    s1 = pd.Series(cat)
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    assert_series_equal(result, pd.Series(['a', 'b', 'a', 'c', 'd']))