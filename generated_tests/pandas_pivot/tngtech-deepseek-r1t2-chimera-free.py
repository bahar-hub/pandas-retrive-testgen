import pytest
import pandas as pd
from pandas import (
    DataFrame,
    Series,
    MultiIndex,
    Categorical,
    date_range,
    Timestamp,
)
from pandas.testing import assert_frame_equal, assert_series_equal
from pandas.core.arrays import BooleanArray


def test_basic_pivot():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    result = df.pivot(index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3], 'two': [2, 4]
    }, index=pd.Index(['bar', 'foo'], name='A'))
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_multi_index_pivot():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['X', 'Y', 'X', 'Y'],
        'C': ['one', 'two', 'one', 'two'],
        'D': [1, 2, 3, 4]
    })
    result = df.pivot(index=['A', 'B'], columns='C', values='D')
    expected = DataFrame({
        'one': [1, 3], 'two': [2, 4]
    }, index=MultiIndex.from_tuples(
        [('bar', 'X'), ('bar', 'Y'), ('foo', 'X'), ('foo', 'Y')][::2],
        names=['A', 'B']
    ))
    expected.columns.name = 'C'
    assert_frame_equal(result, expected)


def test_multi_columns_pivot():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [5, 6, 7, 8]
    })
    result = df.pivot(index='A', columns='B')
    expected = DataFrame({
        ('C', 'one'): [3, 1], ('C', 'two'): [4, 2],
        ('D', 'one'): [7, 5], ('D', 'two'): [8, 6]
    }, index=pd.Index(['bar', 'foo'], name='A'))
    expected.columns.names = [None, 'B']
    assert_frame_equal(result, expected)


def test_values_param_with_list():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [5, 6, 7, 8]
    })
    result = df.pivot(index='A', columns='B', values=['C', 'D'])
    expected = DataFrame({
        ('C', 'one'): [3, 1], ('C', 'two'): [4, 2],
        ('D', 'one'): [7, 5], ('D', 'two'): [8, 6]
    }, index=pd.Index(['bar', 'foo'], name='A'))
    expected.columns.names = [None, 'B']
    assert_frame_equal(result, expected)


def test_no_index_param():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    }).set_index('A')
    result = df.pivot(columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3], 'two': [2, 4]
    }, index=pd.Index(['bar', 'foo'], name='A'))
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_empty_dataframe():
    df = DataFrame(columns=['A', 'B', 'C'])
    result = df.pivot(index='A', columns='B', values='C')
    expected = DataFrame()
    assert_frame_equal(result, expected)


def test_mixed_dtypes():
    df = DataFrame({
        'A': [1, 1, 2, 2],
        'B': ['x', 'y', 'x', 'y'],
        'C': [0.5, 1.5, 2.5, 3.5],
        'D': [True, False, True, False],
        'E': Categorical(['a', 'b', 'a', 'b'])
    })
    result = df.pivot(index='A', columns='B', values=['C', 'D', 'E'])
    expected = DataFrame({
        ('C', 'x'): [2.5, 0.5], ('C', 'y'): [3.5, 1.5],
        ('D', 'x'): [True, True], ('D', 'y'): [False, False],
        ('E', 'x'): ['a', 'a'], ('E', 'y'): ['b', 'b']
    }, index=pd.Index([1, 2], name='A', dtype='int64'))
    expected[('E', 'x')] = Categorical(expected[('E', 'x')], categories=['a', 'b'])
    expected[('E', 'y')] = Categorical(expected[('E', 'y')], categories=['a', 'b'])
    expected.columns.names = [None, 'B']
    assert_frame_equal(result, expected)


def test_duplicate_index_columns_error():
    df = DataFrame({
        'A': ['foo', 'foo', 'foo', 'bar'],
        'B': ['one', 'one', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    with pytest.raises(ValueError, match="Index contains duplicate entries"):
        df.pivot(index='A', columns='B', values='C')


def test_datetime_pivot():
    df = DataFrame({
        'date': date_range('2020-01-01', periods=4),
        'category': ['A', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40]
    })
    result = df.pivot(index='date', columns='category', values='value')
    expected = DataFrame({
        'A': [10, 30], 'B': [20, 40]
    }, index=df['date'].unique())
    expected.index.name = 'date'
    expected.columns.name = 'category'
    assert_frame_equal(result, expected)


def test_boolean_pivot():
    df = DataFrame({
        'A': [True, False, True, False],
        'B': ['x', 'x', 'y', 'y'],
        'C': [1, 2, 3, 4]
    })
    result = df.pivot(index='A', columns='B', values='C')
    expected = DataFrame({
        'x': [2, 1], 'y': [4, 3]
    }, index=pd.Index([False, True], name='A'))
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_series_output():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    result = df.pivot(index='A', columns='B', values='C')
    assert isinstance(result, DataFrame)


def test_multiindex_input():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    }).set_index(['A', 'B'])
    result = df.pivot(columns='B', values='C')
    expected = DataFrame({
        'one': [1, 3], 'two': [2, 4]
    }, index=pd.MultiIndex.from_tuples(
        [('bar', 'one'), ('bar', 'two'), ('foo', 'one'), ('foo', 'two')],
        names=['A', 'B']
    )).unstack()
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_index_names_preservation():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    }).rename_axis('idx_name', axis='index')
    result = df.pivot(index='A', columns='B', values='C')
    assert result.index.name == 'A'
    assert result.columns.name == 'B'


def test_no_values_param():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [5, 6, 7, 8]
    })
    result = df.pivot(index='A', columns='B')
    expected_cols = MultiIndex.from_tuples(
        [('C', 'one'), ('C', 'two'), ('D', 'one'), ('D', 'two')],
        names=[None, 'B']
    )
    expected = DataFrame([
        [3, 4, 7, 8],
        [1, 2, 5, 6]
    ], index=pd.Index(['bar', 'foo'], name='A'), columns=expected_cols)
    assert_frame_equal(result, expected)


def test_view_safety():
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    view = df.pivot(index='A', columns='B', values='C')
    df.loc[0, 'C'] = 100
    assert view.iloc[0, 0] == 5  # Should not change if deep copy was made


def test_categorical_columns():
    df = DataFrame({
        'A': Categorical(['foo', 'foo', 'bar', 'bar'], categories=['bar', 'foo']),
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4]
    })
    result = df.pivot(index='A', columns='B', values='C')
    expected = DataFrame({
        'one': [3, 1], 'two': [4, 2]
    }, index=pd.CategoricalIndex(['bar', 'foo'], categories=['bar', 'foo'], name='A'))
    expected.columns.name = 'B'
    assert_frame_equal(result, expected)


def test_multiindex_columns_result():
    df = DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar'],
        'B': ['one', 'two', 'one', 'two'],
        'C': [1, 2, 3, 4],
        'D': [5, 6, 7, 8]
    })
    result = df.pivot(index='A', columns=['B', 'C'])
    expected_cols = MultiIndex.from_tuples([
        ('D', 'one', 3), ('D', 'two', 4),
        ('D', 'one', 7), ('D', 'two', 8)
    ][::2], names=[None, 'B', 'C'])
    expected = DataFrame(
        [[7, 8], [5, 6]],
        index=pd.Index(['bar', 'foo'], name='A'),
        columns=expected_cols
    )
    assert_frame_equal(result, expected)