import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

def test_series_basic():
    s = pd.Series(['a', 'b', 'a', 'c'])
    result = pd.get_dummies(s)
    expected = pd.DataFrame({
        'a': [True, False, True, False],
        'b': [False, True, False, False],
        'c': [False, False, False, True]
    }, dtype=bool)
    assert_frame_equal(result, expected)

def test_series_with_nan_dummy_na():
    s = pd.Series(['a', np.nan, 'b'])
    result = pd.get_dummies(s, dummy_na=True)
    expected = pd.DataFrame({
        'a': [True, False, False],
        'b': [False, False, True],
        np.nan: [False, True, False]
    }, dtype=bool)
    assert_frame_equal(result, expected)

def test_dataframe_mixed_dtypes():
    df = pd.DataFrame({
        'A': ['x', 'y', 'x'],
        'B': [1, 2, 3],
        'C': pd.Categorical(['a', 'b', 'a'])
    })
    result = pd.get_dummies(df)
    expected = pd.DataFrame({
        'B': [1, 2, 3],
        'A_x': [True, False, True],
        'A_y': [False, True, False],
        'C_a': [True, False, True],
        'C_b': [False, True, False]
    })[['B', 'A_x', 'A_y', 'C_a', 'C_b']]
    assert_frame_equal(result, expected)

def test_columns_parameter():
    df = pd.DataFrame({
        'Encode': ['a', 'b', 'a'],
        'NoEncode': [1, 2, 3],
        'AlsoEncode': ['x', 'x', 'y']
    })
    result = pd.get_dummies(df, columns=['Encode', 'AlsoEncode'])
    expected = pd.DataFrame({
        'NoEncode': [1, 2, 3],
        'Encode_a': [True, False, True],
        'Encode_b': [False, True, False],
        'AlsoEncode_x': [True, True, False],
        'AlsoEncode_y': [False, False, True]
    })
    assert_frame_equal(result, expected)

def test_prefix_str():
    df = pd.DataFrame({'Col': ['a', 'b']})
    result = pd.get_dummies(df, prefix='Pre')
    expected = pd.DataFrame({
        'Pre_a': [True, False],
        'Pre_b': [False, True]
    })
    assert_frame_equal(result, expected)

def test_prefix_list():
    df = pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']})
    result = pd.get_dummies(df, prefix=['PreA', 'PreB'])
    expected = pd.DataFrame({
        'PreA_a': [True, False],
        'PreA_b': [False, True],
        'PreB_c': [True, False],
        'PreB_d': [False, True]
    })
    assert_frame_equal(result, expected)

def test_prefix_dict():
    df = pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']})
    result = pd.get_dummies(df, prefix={'A': 'PreA', 'B': 'PreB'})
    expected = pd.DataFrame({
        'PreA_a': [True, False],
        'PreA_b': [False, True],
        'PreB_c': [True, False],
        'PreB_d': [False, True]
    })
    assert_frame_equal(result, expected)

def test_prefix_sep_variations():
    df = pd.DataFrame({'Col': ['a', 'b']})
    result = pd.get_dummies(df, prefix='Pre', prefix_sep='-')
    expected = pd.DataFrame({
        'Pre-a': [True, False],
        'Pre-b': [False, True]
    })
    assert_frame_equal(result, expected)

def test_drop_first():
    s = pd.Series(['a', 'b', 'a', 'c'])
    result = pd.get_dummies(s, drop_first=True)
    expected = pd.DataFrame({
        'b': [False, True, False, False],
        'c': [False, False, False, True]
    }, dtype=bool)
    assert_frame_equal(result, expected)

def test_dtype_float():
    s = pd.Series(['a', 'b', 'a'])
    result = pd.get_dummies(s, dtype=float)
    expected = pd.DataFrame({
        'a': [1.0, 0.0, 1.0],
        'b': [0.0, 1.0, 0.0]
    })
    assert_frame_equal(result, expected)

def test_sparse_true():
    s = pd.Series(['a', 'b', 'a'])
    result = pd.get_dummies(s, sparse=True)
    assert all(isinstance(col, pd.SparseArray) for col in result.values.T)

def test_empty_series():
    s = pd.Series([], dtype='category')
    result = pd.get_dummies(s)
    expected = pd.DataFrame()
    assert_frame_equal(result, expected)

def test_all_nan_column():
    s = pd.Series([np.nan, np.nan], dtype='category')
    result = pd.get_dummies(s, dummy_na=True)
    expected = pd.DataFrame({np.nan: [True, True]}, dtype=bool)
    assert_frame_equal(result, expected)

def test_single_category_drop_first():
    s = pd.Series(['a', 'a', 'a'])
    result = pd.get_dummies(s, drop_first=True)
    expected = pd.DataFrame(index=[0, 1, 2])
    assert_frame_equal(result, expected)

def test_prefix_length_error():
    df = pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']})
    with pytest.raises(ValueError, match="Length of 'prefix'"):
        pd.get_dummies(df, prefix=['only_one_prefix'])

def test_columns_type_error():
    df = pd.DataFrame({'A': [1, 2]})
    with pytest.raises(TypeError, match="Input must be a list-like"):
        pd.get_dummies(df, columns='not_list_like')

def test_dataframe_no_columns_to_encode():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    result = pd.get_dummies(df)
    assert_frame_equal(result, df)