import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

def test_get_dummies_series():
    s = pd.Series(list('abca'))
    expected = pd.DataFrame({
        'a': [True, False, False, True],
        'b': [False, True, False, False],
        'c': [False, False, True, False]
    })
    result = pd.get_dummies(s)
    assert_frame_equal(result, expected)

def test_get_dummies_series_with_na():
    s = pd.Series(['a', 'b', np.nan])
    expected = pd.DataFrame({
        'a': [True, False, False],
        'b': [False, True, False],
        'NaN': [False, False, True]
    })
    result = pd.get_dummies(s, dummy_na=True)
    assert_frame_equal(result, expected)

def test_get_dummies_series_drop_first():
    s = pd.Series(list('abcaa'))
    expected = pd.DataFrame({
        'b': [False, True, False, False, False],
        'c': [False, False, True, False, False]
    })
    result = pd.get_dummies(s, drop_first=True)
    assert_frame_equal(result, expected)

def test_get_dummies_series_dtype():
    s = pd.Series(list('abc'))
    expected = pd.DataFrame({
        'a': [1.0, 0.0, 0.0],
        'b': [0.0, 1.0, 0.0],
        'c': [0.0, 0.0, 1.0]
    })
    result = pd.get_dummies(s, dtype=float)
    assert_frame_equal(result, expected)

def test_get_dummies_dataframe():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    expected = pd.DataFrame({
        'C': [1, 2, 3],
        'A_a': [True, False, True],
        'A_b': [False, True, False],
        'B_a': [False, True, False],
        'B_b': [True, False, False],
        'B_c': [False, False, True]
    })
    result = pd.get_dummies(df)
    assert_frame_equal(result, expected)

def test_get_dummies_dataframe_with_prefix():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    expected = pd.DataFrame({
        'C': [1, 2, 3],
        'col1_a': [True, False, True],
        'col1_b': [False, True, False],
        'col2_a': [False, True, False],
        'col2_b': [True, False, False],
        'col2_c': [False, False, True]
    })
    result = pd.get_dummies(df, prefix=['col1', 'col2'])
    assert_frame_equal(result, expected)

def test_get_dummies_dataframe_with_prefix_sep():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    expected = pd.DataFrame({
        'C': [1, 2, 3],
        'A-a': [True, False, True],
        'A-b': [False, True, False],
        'B-a': [False, True, False],
        'B-b': [True, False, False],
        'B-c': [False, False, True]
    })
    result = pd.get_dummies(df, prefix_sep='-')
    assert_frame_equal(result, expected)

def test_get_dummies_dataframe_with_columns():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    expected = pd.DataFrame({
        'C': [1, 2, 3],
        'A_a': [True, False, True],
        'A_b': [False, True, False]
    })
    result = pd.get_dummies(df, columns=['A'])
    assert_frame_equal(result, expected)

def test_get_dummies_dataframe_with_sparse():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    result = pd.get_dummies(df, sparse=True)
    assert result['A_a'].dtype == 'sparse[bool]'

def test_get_dummies_empty_series():
    s = pd.Series([])
    expected = pd.DataFrame()
    result = pd.get_dummies(s)
    assert_frame_equal(result, expected)

def test_get_dummies_empty_dataframe():
    df = pd.DataFrame()
    expected = pd.DataFrame()
    result = pd.get_dummies(df)
    assert_frame_equal(result, expected)

def test_get_dummies_with_na():
    s = pd.Series(['a', 'b', np.nan])
    expected = pd.DataFrame({
        'a': [True, False, False],
        'b': [False, True, False]
    })
    result = pd.get_dummies(s)
    assert_frame_equal(result, expected)

def test_get_dummies_with_na_and_dummy_na():
    s = pd.Series(['a', 'b', np.nan])
    expected = pd.DataFrame({
        'a': [True, False, False],
        'b': [False, True, False],
        'NaN': [False, False, True]
    })
    result = pd.get_dummies(s, dummy_na=True)
    assert_frame_equal(result, expected)

def test_get_dummies_with_drop_first():
    s = pd.Series(list('abcaa'))
    expected = pd.DataFrame({
        'b': [False, True, False, False, False],
        'c': [False, False, True, False, False]
    })
    result = pd.get_dummies(s, drop_first=True)
    assert_frame_equal(result, expected)

def test_get_dummies_with_dtype():
    s = pd.Series(list('abc'))
    expected = pd.DataFrame({
        'a': [1.0, 0.0, 0.0],
        'b': [0.0, 1.0, 0.0],
        'c': [0.0, 0.0, 1.0]
    })
    result = pd.get_dummies(s, dtype=float)
    assert_frame_equal(result, expected)

def test_get_dummies_with_prefix_dict():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    expected = pd.DataFrame({
        'C': [1, 2, 3],
        'prefix1_a': [True, False, True],
        'prefix1_b': [False, True, False],
        'prefix2_a': [False, True, False],
        'prefix2_b': [True, False, False],
        'prefix2_c': [False, False, True]
    })
    result = pd.get_dummies(df, prefix={'A': 'prefix1', 'B': 'prefix2'})
    assert_frame_equal(result, expected)

def test_get_dummies_with_prefix_sep_dict():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    expected = pd.DataFrame({
        'C': [1, 2, 3],
        'A-a': [True, False, True],
        'A-b': [False, True, False],
        'B-a': [False, True, False],
        'B-b': [True, False, False],
        'B-c': [False, False, True]
    })
    result = pd.get_dummies(df, prefix_sep={'A': '-', 'B': '-'})
    assert_frame_equal(result, expected)

def test_get_dummies_with_invalid_columns():
    df = pd.DataFrame({
        'A': ['a', 'b', 'a'],
        'B': ['b', 'a', 'c'],
        'C': [1, 2, 3]
    })
    with pytest.raises(TypeError):
        pd.get_dummies(df, columns='A')