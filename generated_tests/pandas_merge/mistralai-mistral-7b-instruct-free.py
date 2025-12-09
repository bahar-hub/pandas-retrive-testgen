import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np

def test_merge_inner_join():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': ['B', 'C'], 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_outer_join():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='outer')
    expected = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value_x': [1, 2, 3, np.nan], 'value_y': [np.nan, 2, 3, 4]})
    pdt.assert_frame_equal(result, expected)

def test_merge_on_different_columns():
    left = pd.DataFrame({'left_key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'right_key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, left_on='left_key', right_on='right_key', how='inner')
    expected = pd.DataFrame({'left_key': ['B', 'C'], 'value_x': [2, 3], 'right_key': ['B', 'C'], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_on_index():
    left = pd.DataFrame({'value': [1, 2, 3]}, index=['A', 'B', 'C'])
    right = pd.DataFrame({'value': [2, 3, 4]}, index=['B', 'C', 'D'])
    result = pd.merge(left, right, left_index=True, right_index=True, how='inner')
    expected = pd.DataFrame({'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_suffix():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner', suffixes=('_left', '_right'))
    expected = pd.DataFrame({'key': ['B', 'C'], 'value_left': [2, 3], 'value_right': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_sort():
    left = pd.DataFrame({'key': ['C', 'A', 'B'], 'value': [3, 1, 2]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner', sort=True)
    expected = pd.DataFrame({'key': ['B', 'C'], 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_indicator():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='outer', indicator=True)
    expected = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value_x': [1, 2, 3, np.nan], 'value_y': [np.nan, 2, 3, 4], '_merge': ['left_only', 'both', 'both', 'right_only']})
    pdt.assert_frame_equal(result, expected)

def test_merge_empty_dataframe():
    left = pd.DataFrame({'key': [], 'value': []})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': [], 'value_x': [], 'value_y': []})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_mixed_dtypes():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2.0, 3.0, 4.0]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': ['B', 'C'], 'value_x': [2, 3], 'value_y': [2.0, 3.0]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_multiindex():
    left = pd.DataFrame({'key1': ['A', 'B', 'C'], 'key2': [1, 2, 3], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key1': ['B', 'C', 'D'], 'key2': [2, 3, 4], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on=['key1', 'key2'], how='inner')
    expected = pd.DataFrame({'key1': ['B', 'C'], 'key2': [2, 3], 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_series():
    left = pd.Series([1, 2, 3], name='value', index=['A', 'B', 'C'])
    right = pd.Series([2, 3, 4], name='value', index=['B', 'C', 'D'])
    result = pd.merge(left, right, left_index=True, right_index=True, how='inner')
    expected = pd.DataFrame({'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_non_unique_index():
    left = pd.DataFrame({'value': [1, 2, 3]}, index=['A', 'B', 'B'])
    right = pd.DataFrame({'value': [2, 3, 4]}, index=['B', 'B', 'D'])
    result = pd.merge(left, right, left_index=True, right_index=True, how='inner')
    expected = pd.DataFrame({'value_x': [2, 2], 'value_y': [2, 2]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_validate():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner', validate='one_to_one')
    expected = pd.DataFrame({'key': ['B', 'C'], 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_datetime():
    left = pd.DataFrame({'key': pd.to_datetime(['2021-01-01', '2021-01-02']), 'value': [1, 2]})
    right = pd.DataFrame({'key': pd.to_datetime(['2021-01-02', '2021-01-03']), 'value': [2, 3]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': pd.to_datetime(['2021-01-02']), 'value_x': [2], 'value_y': [2]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_categorical():
    left = pd.DataFrame({'key': pd.Categorical(['A', 'B', 'C']), 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': pd.Categorical(['B', 'C', 'D']), 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': pd.Categorical(['B', 'C']), 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_bool():
    left = pd.DataFrame({'key': [True, False, True], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': [False, True, False], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': [False, True], 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_object():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': ['B', 'C'], 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)

def test_merge_with_numeric():
    left = pd.DataFrame({'key': [1, 2, 3], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': [2, 3, 4], 'value': [2, 3, 4]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': [2, 3], 'value_x': [2, 3], 'value_y': [2, 3]})
    pdt.assert_frame_equal(result, expected)