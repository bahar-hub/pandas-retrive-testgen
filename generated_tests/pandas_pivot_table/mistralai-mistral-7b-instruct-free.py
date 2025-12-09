import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
        'B': ['one', 'one', 'two', 'two', 'one', 'one'],
        'C': ['x', 'y', 'x', 'y', 'x', 'y'],
        'D': [1, 2, 3, 4, 5, 6]
    })

def test_pivot_table_basic(sample_data):
    result = pd.pivot_table(sample_data, values='D', index='A', columns='B')
    expected = pd.DataFrame({
        'one': {'bar': 5.5, 'foo': 1.5},
        'two': {'bar': 4.0, 'foo': 3.0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_multiple_values(sample_data):
    result = pd.pivot_table(sample_data, values=['D'], index='A', columns='B', aggfunc='sum')
    expected = pd.DataFrame({
        'one': {'bar': 11, 'foo': 3},
        'two': {'bar': 4, 'foo': 3}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_multiple_aggfuncs(sample_data):
    result = pd.pivot_table(sample_data, values='D', index='A', columns='B', aggfunc=['sum', 'mean'])
    expected = pd.DataFrame({
        'sum': {
            'one': {'bar': 11, 'foo': 3},
            'two': {'bar': 4, 'foo': 3}
        },
        'mean': {
            'one': {'bar': 5.5, 'foo': 1.5},
            'two': {'bar': 4.0, 'foo': 3.0}
        }
    })
    assert_frame_equal(result, expected)

def test_pivot_table_fill_value(sample_data):
    result = pd.pivot_table(sample_data, values='D', index='A', columns='C', fill_value=0)
    expected = pd.DataFrame({
        'x': {'bar': 5, 'foo': 2},
        'y': {'bar': 6, 'foo': 2}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_margins(sample_data):
    result = pd.pivot_table(sample_data, values='D', index='A', columns='B', margins=True)
    expected = pd.DataFrame({
        'one': {'bar': 5.5, 'foo': 1.5, 'All': 3.5},
        'two': {'bar': 4.0, 'foo': 3.0, 'All': 3.5},
        'All': {'bar': 9.5, 'foo': 4.5, 'All': 4.0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_dropna_false(sample_data):
    data = sample_data.copy()
    data.loc[0, 'B'] = np.nan
    result = pd.pivot_table(data, values='D', index='A', columns='B', dropna=False)
    expected = pd.DataFrame({
        'nan': {'foo': 1.0},
        'one': {'bar': 5.5, 'foo': 1.5},
        'two': {'bar': 4.0, 'foo': 3.0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_empty_data():
    empty_data = pd.DataFrame()
    result = pd.pivot_table(empty_data, values='D', index='A', columns='B')
    expected = pd.DataFrame()
    assert_frame_equal(result, expected)

def test_pivot_table_no_values():
    data = pd.DataFrame({'A': ['foo', 'bar'], 'B': ['one', 'two']})
    result = pd.pivot_table(data, index='A', columns='B')
    expected = pd.DataFrame({
        'one': {'foo': 0.0, 'bar': 0.0},
        'two': {'foo': 0.0, 'bar': 0.0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_multiple_index_columns(sample_data):
    result = pd.pivot_table(sample_data, values='D', index=['A', 'C'], columns='B')
    expected = pd.DataFrame({
        'one': {
            ('bar', 'x'): 5.0,
            ('bar', 'y'): 6.0,
            ('foo', 'x'): 1.0,
            ('foo', 'y'): 2.0
        },
        'two': {
            ('bar', 'x'): 3.0,
            ('bar', 'y'): 4.0,
            ('foo', 'x'): 3.0,
            ('foo', 'y'): 0.0
        }
    })
    assert_frame_equal(result, expected)

def test_pivot_table_multiple_columns(sample_data):
    result = pd.pivot_table(sample_data, values='D', index='A', columns=['B', 'C'])
    expected = pd.DataFrame({
        ('one', 'x'): {'bar': 5.0, 'foo': 1.0},
        ('one', 'y'): {'bar': 6.0, 'foo': 2.0},
        ('two', 'x'): {'bar': 3.0, 'foo': 3.0},
        ('two', 'y'): {'bar': 4.0, 'foo': 0.0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_sort_false(sample_data):
    result = pd.pivot_table(sample_data, values='D', index='A', columns='B', sort=False)
    expected = pd.DataFrame({
        'one': {'bar': 5.5, 'foo': 1.5},
        'two': {'bar': 4.0, 'foo': 3.0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_observed(sample_data):
    data = sample_data.copy()
    data.loc[0, 'B'] = 'three'
    result = pd.pivot_table(data, values='D', index='A', columns='B', observed=True)
    expected = pd.DataFrame({
        'one': {'bar': 5.5, 'foo': 1.5},
        'two': {'bar': 4.0, 'foo': 3.0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_custom_aggfunc(sample_data):
    def custom_agg(x):
        return x.max() - x.min()
    result = pd.pivot_table(sample_data, values='D', index='A', columns='B', aggfunc=custom_agg)
    expected = pd.DataFrame({
        'one': {'bar': 1, 'foo': 1},
        'two': {'bar': 1, 'foo': 0}
    })
    assert_frame_equal(result, expected)

def test_pivot_table_margins_name(sample_data):
    result = pd.pivot_table(sample_data, values='D', index='A', columns='B', margins=True, margins_name='Total')
    expected = pd.DataFrame({
        'one': {'bar': 5.5, 'foo': 1.5, 'Total': 3.5},
        'two': {'bar': 4.0, 'foo': 3.0, 'Total': 3.5},
        'Total': {'bar': 9.5, 'foo': 4.5, 'Total': 4.0}
    })
    assert_frame_equal(result, expected)