import pytest
import pandas as pd
from pandas import DataFrame, Series, MultiIndex
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np


@pytest.fixture
def basic_df():
    return DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [10, 20, 30, 40]
    })


def test_basic_pivot(basic_df):
    result = pd.pivot(basic_df, index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=['foo', 'bar'])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_no_values(basic_df):
    result = pd.pivot(basic_df, index='A', columns='B')
    expected_data = {
        ('C', 'one'): [1.0, 3.0],
        ('C', 'two'): [2.0, 4.0],
        ('D', 'one'): [10.0, 30.0],
        ('D', 'two'): [20.0, 40.0]
    }
    expected = DataFrame(expected_data, index=['foo', 'bar'])
    expected.columns.names = [None, 'B']
    expected.index.name = 'A'
    assert_frame_equal(result, expected)


def test_pivot_no_index(basic_df):
    result = pd.pivot(basic_df, columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=[0, 2])
    expected.index.name = None
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_only_columns():
    df = DataFrame({'A': [1, 2, 3]})
    result = pd.pivot(df, columns='A')
    expected = DataFrame({
        1: [1.0, np.nan, np.nan],
        2: [np.nan, 2.0, np.nan],
        3: [np.nan, np.nan, 3.0]
    }, index=[0, 1, 2])
    expected.columns.name = 'A'
    assert_frame_equal(result, expected)


def test_pivot_empty_df():
    df = DataFrame(columns=['A', 'B', 'C'])
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame(columns=[])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_single_row():
    df = DataFrame({'A': ['foo'], 'B': ['one'], 'C': [1]})
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame({'one': [1]}, index=['foo'])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_duplicate_index_columns():
    df = DataFrame({
        'A': ['foo', 'foo', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'two'],
        'C': [1, 2, 3, 4]
    })
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [2],
        'two': [4]
    }, index=['foo'])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_multiindex_columns():
    df = DataFrame({
        ('X', 'A'): ['foo', 'foo', 'bar', 'bar'],
        ('X', 'B'): ['one', 'two', 'one', 'two'],
        ('Y', 'C'): [1, 2, 3, 4]
    })
    result = pd.pivot(df, index=('X', 'A'), columns=('X', 'B'), values=('Y', 'C'))
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=['foo', 'bar'])
    expected.index.name = ('X', 'A')
    expected.columns.name = ('X', 'B')
    assert_frame_equal(result, expected)


def test_pivot_multiindex_index():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [10, 20, 30, 40]
    })
    df.index = MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)], names=['L1', 'L2'])
    result = pd.pivot(df, index=['L1', 'L2'], columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=df.index)
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_multiindex_values():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [10, 20, 30, 40]
    })
    result = pd.pivot(df, index='A', columns='B', values=['C', 'D'])
    expected = DataFrame({
        ('C', 'one'): [1, 3],
        ('C', 'two'): [2, 4],
        ('D', 'one'): [10, 30],
        ('D', 'two'): [20, 40]
    }, index=['foo', 'bar'])
    expected.columns.names = [None, 'B']
    expected.index.name = 'A'
    assert_frame_equal(result, expected)


def test_pivot_categorical_columns():
    df = DataFrame({
        'A': pd.Categorical(['foo', 'foo', 'bar', 'bar']),
        'B': pd.Categorical(['one', 'two', 'one', 'two']),
        'C': [1, 2, 3, 4]
    })
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=pd.Categorical(['foo', 'bar']))
    expected.index.name = 'A'
    expected.columns.name = 'B'
    expected.index = expected.index.astype('category')
    assert_frame_equal(result, expected)


def test_pivot_datetime_columns():
    df = DataFrame({
        'A': pd.date_range('2020-01-01', periods=4, freq='D'),
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    result = pd.pivot(df, columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=pd.to_datetime(['2020-01-01', '2020-01-03']))
    expected.index.name = None
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_boolean_values():
    df = DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar'],
        'B': [True, True, False, False],
        'C': [1, 2, 3, 4]
    })
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame({
        False: [3, 4],
        True: [1, 2]
    }, index=['foo', 'bar'])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    expected.columns = expected.columns.astype(bool)
    assert_frame_equal(result, expected)


def test_pivot_numeric_column_names():
    df = DataFrame({
        0: ['foo', 'foo', 'bar', 'bar'],
        1: ['one', 'two', 'one', 'two'],
        2: [1, 2, 3, 4]
    })
    result = pd.pivot(df, index=0, columns=1, values=2)
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=['foo', 'bar'])
    expected.index.name = 0
    expected.columns.name = 1
    assert_frame_equal(result, expected)


def test_pivot_none_column_name():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    df.columns = ['A', None, 'C']
    result = pd.pivot(df, index='A', columns=None, values='C')
    expected = DataFrame({
        'one': [1, 3],
        'two': [2, 4]
    }, index=['foo', 'bar'])
    expected.index.name = 'A'
    expected.columns.name = None
    assert_frame_equal(result, expected)


def test_pivot_mixed_dtypes():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': [1, 2, 1, 2],
        'C': [1.0, 2.0, 3.0, 4.0],
        'D': [True, False, True, False]
    })
    result = pd.pivot(df, index='A', columns='B', values=['C', 'D'])
    expected = DataFrame({
        ('C', 1): [1.0, 3.0],
        ('C', 2): [2.0, 4.0],
        ('D', 1): [True, True],
        ('D', 2): [False, False]
    }, index=['foo', 'bar'])
    expected.columns.names = [None, 'B']
    expected.index.name = 'A'
    assert_frame_equal(result, expected)


def test_pivot_no_values_no_index():
    df = DataFrame({
        'A': ['foo', 'bar'],
        'B': ['one', 'two']
    })
    result = pd.pivot(df, columns='B')
    expected = DataFrame({
        ('A', 'one'): ['foo', np.nan],
        ('A', 'two'): [np.nan, 'bar']
    }, index=[0, 1])
    expected.columns.names = [None, 'B']
    assert_frame_equal(result, expected)


def test_pivot_with_nan_values():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, np.nan, 3, 4]
    })
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [1.0, 3.0],
        'two': [np.nan, 4.0]
    }, index=['foo', 'bar'])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_all_nan_values():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [np.nan, np.nan, np.nan, np.nan]
    })
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [np.nan, np.nan],
        'two': [np.nan, np.nan]
    }, index=['foo', 'bar'])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_duplicate_combinations_last_value_taken():
    df = DataFrame({
        'A': ['foo', 'foo', 'foo'],
        'B': ['one', 'one', 'one'],
        'C': [1, 2, 3]
    })
    result = pd.pivot(df, index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [3]
    }, index=['foo'])
    expected.index.name = 'A'
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_pivot_series_result():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    result = pd.pivot(df, index='A', columns='B', values='C')
    assert isinstance(result, DataFrame)


def test_pivot_columns_none():
    df = DataFrame({
        'A': ['foo', 'bar'],
        'B': [1, 2]
    })
    result = pd.pivot(df, columns=None, values='B')
    expected = DataFrame({
        'foo': [1.0, np.nan],
        'bar': [np.nan, 2.0]
    }, index=[0, 1])
    expected.columns.name = None
    assert_frame_equal(result, expected)