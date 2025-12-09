import pandas as pd
import pytest

def test_pivot_table_empty_data():
    data = pd.DataFrame()
    with pytest.raises(ValueError):
        pd.pivot_table(data)

def test_pivot_table_no_index_or_columns():
    data = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(ValueError):
        pd.pivot_table(data)

def test_pivot_table_single_index():
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.pivot_table(data, index='A', values='B')
    expected = pd.DataFrame({'B': [4, 5, 6]}, index=[1, 2, 3])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_single_index_and_columns():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': ['x', 'y', 'x', 'y'], 'C': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', columns='B', values='C')
    expected = pd.DataFrame({'x': [4, 6], 'y': [5, 7]}, index=[1, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_multiple_index():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 4, 5, 5], 'C': [6, 7, 8, 9]})
    result = pd.pivot_table(data, index=['A', 'B'], values='C')
    expected = pd.DataFrame({'C': [6, 7, 8, 9]}, index=pd.MultiIndex.from_tuples([(1, 4), (1, 4), (2, 5), (2, 5)], names=['A', 'B']))
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_aggfunc_mean():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', aggfunc='mean')
    expected = pd.DataFrame({'B': [4.5, 6.5]}, index=[1, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_aggfunc_sum():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', aggfunc='sum')
    expected = pd.DataFrame({'B': [9, 13]}, index=[1, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_aggfunc_list():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', aggfunc=['mean', 'sum'])
    expected = pd.DataFrame({'mean': [4.5, 6.5], 'sum': [9, 13]}, index=[1, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_margins():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', margins=True)
    expected = pd.DataFrame({'B': [4.5, 6.5, 5.5]}, index=[1, 2, 'All'])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_margins_name():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', margins=True, margins_name='Total')
    expected = pd.DataFrame({'B': [4.5, 6.5, 5.5]}, index=[1, 2, 'Total'])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_dropna():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, None, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', dropna=False)
    expected = pd.DataFrame({'B': [4.0, None, 6.5, 7.0]}, index=[1, 1, 2, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_observed():
    data = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', observed=True)
    expected = pd.DataFrame({'B': [4.5, 6.5]}, index=[1, 2])
    pd.testing.assert_frame_equal(result, expected)

def test_pivot_table_sort():
    data = pd.DataFrame({'A': [2, 2, 1, 1], 'B': [4, 5, 6, 7]})
    result = pd.pivot_table(data, index='A', values='B', sort=False)
    expected = pd.DataFrame({'B': [6.5, 4.5]}, index=[2, 1])
    pd.testing.assert_frame_equal(result, expected)