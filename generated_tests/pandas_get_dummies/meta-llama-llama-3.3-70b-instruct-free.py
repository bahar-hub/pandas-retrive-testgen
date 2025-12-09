import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

def test_get_dummies_series():
    s = pd.Series(list('abca'))
    expected = pd.DataFrame({'a': [True, False, False, True], 'b': [False, True, False, False], 'c': [False, False, True, False]})
    assert_frame_equal(pd.get_dummies(s), expected)

def test_get_dummies_series_with_prefix():
    s = pd.Series(list('abca'))
    expected = pd.DataFrame({'x_a': [True, False, False, True], 'x_b': [False, True, False, False], 'x_c': [False, False, True, False]})
    assert_frame_equal(pd.get_dummies(s, prefix='x'), expected)

def test_get_dummies_series_with_prefix_sep():
    s = pd.Series(list('abca'))
    expected = pd.DataFrame({'a-x': [True, False, False, True], 'b-x': [False, True, False, False], 'c-x': [False, False, True, False]})
    assert_frame_equal(pd.get_dummies(s, prefix='x', prefix_sep='-'), expected)

def test_get_dummies_series_with_dummy_na():
    s = pd.Series(['a', 'b', np.nan])
    expected = pd.DataFrame({'a': [True, False, False], 'b': [False, True, False]})
    assert_frame_equal(pd.get_dummies(s), expected)

def test_get_dummies_series_with_dummy_na_true():
    s = pd.Series(['a', 'b', np.nan])
    expected = pd.DataFrame({'a': [True, False, False], 'b': [False, True, False], 'NaN': [False, False, True]})
    assert_frame_equal(pd.get_dummies(s, dummy_na=True), expected)

def test_get_dummies_series_with_sparse():
    s = pd.Series(list('abca'))
    expected = pd.DataFrame({'a': [True, False, False, True], 'b': [False, True, False, False], 'c': [False, False, True, False]})
    assert_frame_equal(pd.get_dummies(s, sparse=True), expected)

def test_get_dummies_series_with_drop_first():
    s = pd.Series(list('abca'))
    expected = pd.DataFrame({'b': [False, True, False, False], 'c': [False, False, True, False]})
    assert_frame_equal(pd.get_dummies(s, drop_first=True), expected)

def test_get_dummies_series_with_dtype():
    s = pd.Series(list('abca'))
    expected = pd.DataFrame({'a': [1.0, 0.0, 0.0, 1.0], 'b': [0.0, 1.0, 0.0, 0.0], 'c': [0.0, 0.0, 1.0, 0.0]})
    assert_frame_equal(pd.get_dummies(s, dtype=float), expected)

def test_get_dummies_dataframe():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'A_a': [True, False, True], 'A_b': [False, True, False], 'B_a': [False, True, False], 'B_b': [True, False, False], 'B_c': [False, False, True]})
    assert_frame_equal(pd.get_dummies(df), expected)

def test_get_dummies_dataframe_with_prefix():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'x_a': [True, False, True], 'x_b': [False, True, False], 'y_a': [False, True, False], 'y_b': [True, False, False], 'y_c': [False, False, True]})
    assert_frame_equal(pd.get_dummies(df, prefix=['x', 'y']), expected)

def test_get_dummies_dataframe_with_prefix_sep():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'A-x': [True, False, True], 'A-y': [False, True, False], 'B-x': [False, True, False], 'B-y': [True, False, False], 'B-z': [False, False, True]})
    assert_frame_equal(pd.get_dummies(df, prefix=['x', 'y'], prefix_sep=['-', '-']), expected)

def test_get_dummies_dataframe_with_dummy_na():
    df = pd.DataFrame({'A': ['a', 'b', np.nan], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'A_a': [True, False, False], 'A_b': [False, True, False], 'B_a': [False, True, False], 'B_b': [True, False, False], 'B_c': [False, False, True]})
    assert_frame_equal(pd.get_dummies(df), expected)

def test_get_dummies_dataframe_with_dummy_na_true():
    df = pd.DataFrame({'A': ['a', 'b', np.nan], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'A_a': [True, False, False], 'A_b': [False, True, False], 'A_NaN': [False, False, True], 'B_a': [False, True, False], 'B_b': [True, False, False], 'B_c': [False, False, True]})
    assert_frame_equal(pd.get_dummies(df, dummy_na=True), expected)

def test_get_dummies_dataframe_with_sparse():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'A_a': [True, False, True], 'A_b': [False, True, False], 'B_a': [False, True, False], 'B_b': [True, False, False], 'B_c': [False, False, True]})
    assert_frame_equal(pd.get_dummies(df, sparse=True), expected)

def test_get_dummies_dataframe_with_drop_first():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'A_b': [False, True, False], 'B_a': [False, True, False], 'B_b': [True, False, False], 'B_c': [False, False, True]})
    assert_frame_equal(pd.get_dummies(df, drop_first=True), expected)

def test_get_dummies_dataframe_with_dtype():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    expected = pd.DataFrame({'C': [1, 2, 3], 'A_a': [1.0, 0.0, 1.0], 'A_b': [0.0, 1.0, 0.0], 'B_a': [0.0, 1.0, 0.0], 'B_b': [1.0, 0.0, 0.0], 'B_c': [0.0, 0.0, 1.0]})
    assert_frame_equal(pd.get_dummies(df, dtype=float), expected)

def test_get_dummies_invalid_columns():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    with pytest.raises(TypeError):
        pd.get_dummies(df, columns='invalid')

def test_get_dummies_invalid_prefix():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    with pytest.raises(ValueError):
        pd.get_dummies(df, prefix='x', prefix_sep='y', columns=['A', 'B', 'C'])

def test_get_dummies_invalid_prefix_sep():
    df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    with pytest.raises(ValueError):
        pd.get_dummies(df, prefix='x', prefix_sep='y', columns=['A', 'B', 'C'])