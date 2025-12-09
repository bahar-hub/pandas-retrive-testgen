import pandas as pd
import pytest
from pandas.core.reshape import concat
from itertools import cycle
import numpy as np

def _get_dummies_1d(data, prefix=None, prefix_sep='_', dummy_na=False, sparse=False, drop_first=False, dtype=bool):
    """Helper function to get dummies for a 1D array-like object."""
    if dummy_na:
        return pd.get_dummies(data, prefix=prefix, prefix_sep=prefix_sep, dummy_na=True, sparse=sparse, drop_first=drop_first, dtype=dtype)
    else:
        return pd.get_dummies(data, prefix=prefix, prefix_sep=prefix_sep, sparse=sparse, drop_first=drop_first, dtype=dtype)

def test_empty_series():
    s = pd.Series([])
    result = _get_dummies_1d(s)
    pd.testing.assert_frame_equal(result, pd.DataFrame(), check_dtype=False)

def test_empty_dataframe():
    df = pd.DataFrame()
    result = _get_dummies_1d(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(), check_dtype=False)

def test_single_value_series():
    s = pd.Series(['a'])
    result = _get_dummies_1d(s)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'a': [True, False]}, index=['a', 'b']), check_dtype=False)

def test_single_value_dataframe():
    df = pd.DataFrame({'A': ['a']})
    result = _get_dummies_1d(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'A': ['a']}, index=['A']), check_dtype=False)

def test_simple_series():
    s = pd.Series(['a', 'b', 'c'])
    result = _get_dummies_1d(s)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'a': [True, False, False],
                                                     'b': [False, True, False],
                                                     'c': [False, False, True]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_simple_dataframe():
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['b', 'a', 'c']})
    result = _get_dummies_1d(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'A': [True, False, False],
                                                     'B': [False, True, False]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_with_nan_series():
    s = pd.Series(['a', 'b', np.nan])
    result = _get_dummies_1d(s)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'a': [True, False, False],
                                                     'b': [False, True, False]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_with_nan_dataframe():
    df = pd.DataFrame({'A': ['a', 'b', np.nan], 'B': ['b', 'a', np.nan]})
    result = _get_dummies_1d(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'A': [True, False, False],
                                                     'B': [False, True, False]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_with_prefix():
    s = pd.Series(['a', 'b', 'c'])
    result = _get_dummies_1d(s, prefix='prefix_')
    pd.testing.assert_frame_equal(result, pd.DataFrame({'prefix_a': [True, False, False],
                                                     'prefix_b': [False, True, False],
                                                     'prefix_c': [False, False, True]},
                                    index=['prefix_a', 'prefix_b', 'prefix_c']), check_dtype=False)

def test_with_prefix_and_separator():
    s = pd.Series(['a', 'b', 'c'])
    result = _get_dummies_1d(s, prefix='prefix', prefix_sep='_')
    pd.testing.assert_frame_equal(result, pd.DataFrame({'prefix_a': [True, False, False],
                                                     'prefix_b': [False, True, False],
                                                     'prefix_c': [False, False, True]},
                                    index=['prefix_a', 'prefix_b', 'prefix_c']), check_dtype=False)

def test_with_prefix_and_dict():
    s = pd.Series(['a', 'b', 'c'])
    result = _get_dummies_1d(s, prefix={'a': 'prefix_a', 'b': 'prefix_b'}, prefix_sep='_')
    pd.testing.assert_frame_equal(result, pd.DataFrame({'prefix_a': [True, False, False],
                                                     'prefix_b': [False, True, False],
                                                     'prefix_c': [False, False, True]},
                                    index=['prefix_a', 'prefix_b', 'prefix_c']), check_dtype=False)

def test_drop_first():
    s = pd.Series(['a', 'b', 'c'])
    result = _get_dummies_1d(s, drop_first=True)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'a': [True, False, False],
                                                     'b': [False, True, False],
                                                     'c': [False, False, True]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_dtype():
    s = pd.Series(['a', 'b', 'c'])
    result = _get_dummies_1d(s, dtype=float)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'a': [1.0, 0.0, 0.0],
                                                     'b': [0.0, 1.0, 0.0],
                                                     'c': [0.0, 0.0, 1.0]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_sparse():
    s = pd.Series(['a', 'b', 'c'])
    result = _get_dummies_1d(s, sparse=True)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'a': [True, False, False],
                                                     'b': [False, True, False],
                                                     'c': [False, False, True]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_columns_parameter():
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    result = _get_dummies_1d(df, columns=['A', 'B'])
    pd.testing.assert_frame_equal(result, pd.DataFrame({'A': [True, False, False],
                                                     'B': [False, True, False]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_columns_parameter_all():
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
    result = _get_dummies_1d(df, columns=None)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'A': [True, False, False],
                                                     'B': [False, True, False],
                                                     'C': [False, False, True]},
                                    index=['a', 'b', 'c']), check_dtype=False)

def test_mixed_dtypes():
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 2, 3], 'C': ['x', 'y', 'z']})
    result = _get_dummies_1d(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({'A': [True, False, False],
                                                     'B': [False, True, False],
                                                     'C': [False, False, True]},
                                    index=['a', 'b', 'c']), check_dtype=False)