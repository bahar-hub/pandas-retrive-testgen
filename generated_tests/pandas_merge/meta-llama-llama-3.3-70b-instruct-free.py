import pandas as pd
import pandas.testing as pd_testing
import pytest

def test_merge_inner():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': ['A', 'B'], 'value_x': [1, 2], 'value_y': [4, 5]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_outer():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='outer')
    expected = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value_x': [1, 2, 3, None], 'value_y': [4, 5, None, 6]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_left_index():
    left = pd.DataFrame({'value': [1, 2, 3]}, index=['A', 'B', 'C'])
    right = pd.DataFrame({'value': [4, 5, 6]}, index=['A', 'B', 'D'])
    result = pd.merge(left, right, left_index=True, right_index=True, how='inner')
    expected = pd.DataFrame({'value_x': [1, 2], 'value_y': [4, 5]}, index=['A', 'B'])
    pd_testing.assert_frame_equal(result, expected)

def test_merge_right_index():
    left = pd.DataFrame({'value': [1, 2, 3]}, index=['A', 'B', 'C'])
    right = pd.DataFrame({'value': [4, 5, 6]}, index=['A', 'B', 'D'])
    result = pd.merge(left, right, left_index=True, right_index=True, how='outer')
    expected = pd.DataFrame({'value_x': [1, 2, None], 'value_y': [4, 5, 6]}, index=['A', 'B', 'D'])
    pd_testing.assert_frame_equal(result, expected)

def test_merge_sort():
    left = pd.DataFrame({'key': ['B', 'A', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner', sort=True)
    expected = pd.DataFrame({'key': ['A', 'B'], 'value_x': [2, 1], 'value_y': [4, 5]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_suffixes():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner', suffixes=('_left', '_right'))
    expected = pd.DataFrame({'key': ['A', 'B'], 'value_left': [1, 2], 'value_right': [4, 5]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_indicator():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='outer', indicator=True)
    expected = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value_x': [1, 2, 3, None], 'value_y': [4, 5, None, 6], '_merge': ['both', 'both', 'left_only', 'right_only']})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_validate():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    with pytest.raises(ValueError):
        pd.merge(left, right, on='key', how='inner', validate='1:1')

def test_merge_empty():
    left = pd.DataFrame({'key': [], 'value': []})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': [], 'value_x': [], 'value_y': []})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_mixed_dtypes():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4.0, 5.0, 6.0]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': ['A', 'B'], 'value_x': [1, 2], 'value_y': [4.0, 5.0]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_non_unique_index():
    left = pd.DataFrame({'value': [1, 2, 3]}, index=['A', 'A', 'B'])
    right = pd.DataFrame({'value': [4, 5, 6]}, index=['A', 'B', 'D'])
    result = pd.merge(left, right, left_index=True, right_index=True, how='inner')
    expected = pd.DataFrame({'value_x': [1, 2], 'value_y': [4, 5]}, index=['A', 'A'])
    pd_testing.assert_frame_equal(result, expected)

def test_merge_multiindex():
    left = pd.DataFrame({'value': [1, 2, 3]}, index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['key1', 'key2']))
    right = pd.DataFrame({'value': [4, 5, 6]}, index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['key1', 'key2']))
    result = pd.merge(left, right, left_index=True, right_index=True, how='inner')
    expected = pd.DataFrame({'value_x': [1, 2, 3], 'value_y': [4, 5, 6]}, index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['key1', 'key2']))
    pd_testing.assert_frame_equal(result, expected)

def test_merge_datetime():
    left = pd.DataFrame({'key': pd.date_range('2022-01-01', periods=3), 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': pd.date_range('2022-01-01', periods=3), 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': pd.date_range('2022-01-01', periods=3), 'value_x': [1, 2, 3], 'value_y': [4, 5, 6]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_categorical():
    left = pd.DataFrame({'key': pd.Categorical(['A', 'B', 'C']), 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': pd.Categorical(['A', 'B', 'D']), 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': pd.Categorical(['A', 'B']), 'value_x': [1, 2], 'value_y': [4, 5]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_bool():
    left = pd.DataFrame({'key': [True, False, True], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': [True, False, True], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': [True, False], 'value_x': [1, 2], 'value_y': [4, 5]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_object():
    left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': ['A', 'B'], 'value_x': [1, 2], 'value_y': [4, 5]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_numeric():
    left = pd.DataFrame({'key': [1, 2, 3], 'value': [1.0, 2.0, 3.0]})
    right = pd.DataFrame({'key': [1, 2, 4], 'value': [4.0, 5.0, 6.0]})
    result = pd.merge(left, right, on='key', how='inner')
    expected = pd.DataFrame({'key': [1, 2], 'value_x': [1.0, 2.0], 'value_y': [4.0, 5.0]})
    pd_testing.assert_frame_equal(result, expected)

def test_merge_series():
    left = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
    right = pd.Series([4, 5, 6], index=['A', 'B', 'D'])
    result = pd.merge(left.to_frame('value'), right.to_frame('value'), left_index=True, right_index=True, how='inner')
    expected = pd.DataFrame({'value_x': [1, 2], 'value_y': [4, 5]}, index=['A', 'B'])
    pd_testing.assert_frame_equal(result, expected)