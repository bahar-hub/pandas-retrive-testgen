import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
        'B': ['one', 'one', 'two', 'two', 'one', 'one'],
        'C': ['x', 'y', 'x', 'y', 'x', 'y'],
        'D': [1, 2, 3, 4, 5, 6],
        'E': [10, 20, 30, 40, 50, 60]
    })

def test_pivot_basic(sample_df):
    result = sample_df.pivot(index='A', columns='B', values='D')
    expected = pd.DataFrame({
        'one': [1, 5],
        'two': [3, 4]
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_multiple_values(sample_df):
    result = sample_df.pivot(index='A', columns='B', values=['D', 'E'])
    expected = pd.DataFrame({
        ('D', 'one'): [1, 5],
        ('D', 'two'): [3, 4],
        ('E', 'one'): [10, 50],
        ('E', 'two'): [30, 40]
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_no_index(sample_df):
    result = sample_df.pivot(columns='B', values='D')
    expected = pd.DataFrame({
        'one': [1, 2, 5, 6],
        'two': [3, 4]
    }, index=[0, 1, 3, 4])
    pdt.assert_frame_equal(result, expected)

def test_pivot_no_values(sample_df):
    result = sample_df.pivot(index='A', columns='B')
    expected = pd.DataFrame({
        'C': {'one': ['x', 'x'], 'two': ['x', 'y']},
        'D': {'one': [1, 5], 'two': [3, 4]},
        'E': {'one': [10, 50], 'two': [30, 40]}
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_empty_df():
    df = pd.DataFrame()
    result = df.pivot(index='A', columns='B', values='C')
    expected = pd.DataFrame()
    pdt.assert_frame_equal(result, expected)

def test_pivot_mixed_dtypes():
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [1.1, 2.2, 3.3, 4.4],
        'E': [True, False, True, False]
    })
    result = df.pivot(index='A', columns='B', values=['C', 'D', 'E'])
    expected = pd.DataFrame({
        ('C', 'one'): [1, 3],
        ('C', 'two'): [2, 4],
        ('D', 'one'): [1.1, 3.3],
        ('D', 'two'): [2.2, 4.4],
        ('E', 'one'): [True, True],
        ('E', 'two'): [False, False]
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_datetime():
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
    })
    result = df.pivot(index='A', columns='B', values='C')
    expected = pd.DataFrame({
        'one': pd.to_datetime(['2020-01-01', '2020-01-03']),
        'two': pd.to_datetime(['2020-01-02', '2020-01-04'])
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_categorical():
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': pd.Categorical(['a', 'b', 'a', 'b'])
    })
    result = df.pivot(index='A', columns='B', values='C')
    expected = pd.DataFrame({
        'one': pd.Categorical(['a', 'a']),
        'two': pd.Categorical(['b', 'b'])
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_bool():
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [True, False, True, False]
    })
    result = df.pivot(index='A', columns='B', values='C')
    expected = pd.DataFrame({
        'one': [True, True],
        'two': [False, False]
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_non_unique_index():
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
        'B': ['one', 'one', 'two', 'two', 'one', 'one'],
        'C': [1, 2, 3, 4, 5, 6]
    })
    with pytest.raises(ValueError):
        df.pivot(index='A', columns='B', values='C')

def test_pivot_multiindex():
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [10, 20, 30, 40]
    }).set_index(['A', 'B'])
    result = df.pivot(columns='A')
    expected = pd.DataFrame({
        'C': {'foo': [1, 2], 'bar': [3, 4]},
        'D': {'foo': [10, 20], 'bar': [30, 40]}
    }, index=['one', 'two'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_series():
    s = pd.Series([1, 2, 3, 4], index=['foo', 'foo', 'bar', 'bar'], name='values')
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two']
    })
    result = df.pivot(index='A', columns='B').assign(values=s)
    expected = pd.DataFrame({
        'values': {'one': [1, 3], 'two': [2, 4]}
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_join_outer():
    df1 = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    df2 = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'D': [10, 20, 30, 40]
    })
    result = df1.pivot(index='A', columns='B', values='C').join(
        df2.pivot(index='A', columns='B', values='D')
    )
    expected = pd.DataFrame({
        'C': {'one': [1, 3], 'two': [2, 4]},
        'D': {'one': [10, 30], 'two': [20, 40]}
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)

def test_pivot_join_inner():
    df1 = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    df2 = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'D': [10, 20, 30, 40]
    })
    result = df1.pivot(index='A', columns='B', values='C').join(
        df2.pivot(index='A', columns='B', values='D'), how='inner'
    )
    expected = pd.DataFrame({
        'C': {'one': [1, 3], 'two': [2, 4]},
        'D': {'one': [10, 30], 'two': [20, 40]}
    }, index=['foo', 'bar'])
    pdt.assert_frame_equal(result, expected)