import pytest
import pandas as pd
from pandas import DataFrame, Series, MultiIndex, merge
from pandas.testing import assert_frame_equal, assert_series_equal
from datetime import datetime

def test_inner_join_basic():
    left = DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = DataFrame({'key': ['B', 'C', 'D'], 'value': [4, 5, 6]})
    result = merge(left, right, on='key', how='inner')
    expected = DataFrame({'key': ['B', 'C'], 'value_x': [2, 3], 'value_y': [4, 5]})
    assert_frame_equal(result, expected)

def test_outer_join_sort():
    left = DataFrame({'key': [3, 1], 'val': ['a', 'b']})
    right = DataFrame({'key': [2, 1], 'val': ['c', 'd']})
    result = merge(left, right, on='key', how='outer', sort=True)
    expected = DataFrame({'key': [1, 2, 3], 'val_x': ['b', None, 'a'], 'val_y': ['d', 'c', None]})
    assert_frame_equal(result, expected)

def test_merge_on_index():
    left = DataFrame({'value': [10, 20]}, index=['X', 'Y'])
    right = DataFrame({'value': [30, 40]}, index=['Y', 'Z'])
    result = merge(left, right, left_index=True, right_index=True, how='left', suffixes=('_l', '_r'))
    expected = DataFrame({'value_l': [10, 20], 'value_r': [None, 30]}, index=['X', 'Y'])
    assert_frame_equal(result, expected)

def test_merge_mixed_keys():
    left = DataFrame({'key': [1, 2], 'val': ['a', 'b']}, index=[10, 20])
    right = DataFrame({'key': [2, 3], 'val': ['c', 'd']})
    result = merge(left, right, left_index=True, right_on='key', how='inner')
    expected = DataFrame({'key_x': [2], 'val_x': ['b'], 'key_y': [2], 'val_y': ['c']}, index=pd.Index([20], dtype='int64'))
    assert_frame_equal(result, expected)

def test_empty_left():
    left = DataFrame(columns=['key', 'val'])
    right = DataFrame({'key': [1, 2], 'val': ['a', 'b']})
    result = merge(left, right, on='key', how='outer')
    expected = DataFrame({'key': [1, 2], 'val_x': [None, None], 'val_y': ['a', 'b']}, dtype=object)
    assert_frame_equal(result, expected)

def test_both_empty():
    left = DataFrame(columns=['key'])
    right = DataFrame(columns=['key'])
    result = merge(left, right, on='key')
    assert result.empty

def test_dtype_mix():
    left = DataFrame({
        'key': [1, 2],
        'dt': [datetime(2020,1,1), datetime(2020,1,2)],
        'cat': pd.Categorical(['x', 'y']),
        'bool': [True, False]
    })
    right = DataFrame({
        'key': [2, 3],
        'num': [1.5, 2.5],
        'obj': ['a', 'b']
    })
    result = merge(left, right, on='key', how='outer')
    expected = DataFrame({
        'key': [1, 2, 3],
        'dt': [datetime(2020,1,1), datetime(2020,1,2), None],
        'cat': pd.Categorical(['x', 'y', None], categories=['x', 'y']),
        'bool': [True, False, None],
        'num': [None, 1.5, 2.5],
        'obj': [None, 'a', 'b']
    })
    assert_frame_equal(result, expected)

def test_non_unique_index():
    left = DataFrame({'key': [1, 1], 'val': ['a', 'b']}).set_index('key')
    right = DataFrame({'key': [1, 1], 'val': ['c', 'd']}).set_index('key')
    result = merge(left, right, left_index=True, right_index=True, suffixes=('_l', '_r'))
    expected = DataFrame({
        'val_l': ['a', 'a', 'b', 'b'],
        'val_r': ['c', 'd', 'c', 'd']
    }, index=pd.Index([1, 1, 1, 1], name='key'))
    assert_frame_equal(result, expected)

def test_multiindex_columns():
    left = DataFrame({('A', 'key'): [1, 2], ('B', 'val'): ['a', 'b']})
    right = DataFrame({('A', 'key'): [2, 3], ('C', 'val'): ['c', 'd']})
    result = merge(left, right, on=[('A', 'key')])
    expected = DataFrame({
        ('A', 'key'): [2],
        ('B', 'val'): ['b'],
        ('C', 'val'): ['c']
    })
    assert_frame_equal(result, expected)

def test_series_merge():
    s = Series([10, 20], name='val', index=[1, 2])
    df = DataFrame({'key': [1, 3], 'data': ['a', 'b']})
    result = merge(s, df, left_index=True, right_on='key', how='outer')
    expected = DataFrame({
        'key': [1, 2, 3],
        'val': [10.0, 20.0, None],
        'data': ['a', None, 'b']
    })
    assert_frame_equal(result, expected)

def test_validate_one_to_one():
    left = DataFrame({'key': [1, 2]})
    right = DataFrame({'key': [2, 3]})
    merge(left, right, on='key', how='inner', validate='one_to_one')

def test_validate_raises():
    left = DataFrame({'key': [1, 1]})
    right = DataFrame({'key': [1]})
    with pytest.raises(pd.errors.MergeError):
        merge(left, right, on='key', validate='one_to_one')

def test_indicator():
    left = DataFrame({'key': [1, 2]})
    right = DataFrame({'key': [2, 3]})
    result = merge(left, right, on='key', how='outer', indicator=True)
    expected = DataFrame({
        'key': [1, 2, 3],
        '_merge': ['left_only', 'both', 'right_only']
    })
    assert_frame_equal(result, expected)

def test_suffixes():
    left = DataFrame({'key': [1], 'val': [10]})
    right = DataFrame({'key': [1], 'val': [20]})
    result = merge(left, right, on='key', suffixes=('_left', '_right'))
    expected = DataFrame({'key': [1], 'val_left': [10], 'val_right': [20]})
    assert_frame_equal(result, expected)

def test_copy_no_side_effects():
    left = DataFrame({'key': [1]})
    right = DataFrame({'key': [1]})
    left_orig = left.copy()
    merge(left, right, on='key', copy=True)
    assert_frame_equal(left, left_orig)

def test_ignore_index():
    left = DataFrame({'key': [1]}, index=[10])
    right = DataFrame({'key': [1]}, index=[20])
    result = merge(left, right, on='key', ignore_index=True)
    assert result.index.equals(pd.RangeIndex(start=0, stop=1))

def test_multiindex_merge():
    left = DataFrame({'a': [1, 2], 'b': [3, 4], 'val': ['x', 'y']}).set_index(['a', 'b'])
    right = DataFrame({'a': [2, 3], 'b': [4, 5], 'val': ['z', 'w']}).set_index(['a', 'b'])
    result = merge(left, right, left_index=True, right_index=True, suffixes=('_l', '_r'))
    expected = DataFrame({
        'val_l': ['y'],
        'val_r': ['z']
    }, index=MultiIndex.from_tuples([(2, 4)], names=['a', 'b']))
    assert_frame_equal(result, expected)

def test_categorical_keys():
    left = DataFrame({'key': pd.Categorical(['a', 'b']), 'val': [1, 2]})
    right = DataFrame({'key': pd.Categorical(['b', 'c']), 'val': [3, 4]})
    result = merge(left, right, on='key', how='outer')
    expected = DataFrame({
        'key': pd.Categorical(['a', 'b', 'c']),
        'val_x': [1, 2, None],
        'val_y': [None, 3, 4]
    })
    assert_frame_equal(result, expected)

def test_boolean_dtype():
    left = DataFrame({'key': [True, False], 'val': [1, 2]})
    right = DataFrame({'key': [False, None], 'val': [3, 4]})
    result = merge(left, right, on='key', how='outer')
    expected = DataFrame({
        'key': [True, False, None],
        'val_x': [1, 2, None],
        'val_y': [None, 3, 4]
    }, dtype=object)
    assert_frame_equal(result, expected)

def test_verify_integrity_raises():
    left = DataFrame({'key': [1, 1]})
    right = DataFrame({'key': [1]})
    with pytest.raises(pd.errors.MergeError):
        merge(left, right, on='key', verify_integrity=True)

def test_merge_series_with_name():
    s = Series([1, 2], name='val')
    df = DataFrame({'val': [2, 3], 'data': ['a', 'b']})
    result = merge(s, df, left_on='val', right_on='val')
    expected = DataFrame({'val': [2], 'data': ['a']})
    assert_frame_equal(result, expected)