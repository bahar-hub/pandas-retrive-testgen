import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

def test_crosstab_basic():
    index = ['foo', 'foo', 'foo', 'bar', 'bar', 'bar']
    columns = ['one', 'one', 'two', 'one', 'two', 'two']
    result = pd.crosstab(index, columns)
    expected = pd.DataFrame({'one': [2, 1], 'two': [1, 2]}, index=['bar', 'foo'])
    assert_frame_equal(result, expected)

def test_crosstab_with_values_and_aggfunc():
    index = ['foo', 'foo', 'bar', 'bar']
    columns = ['one', 'two', 'one', 'two']
    values = [1, 2, 3, 4]
    result = pd.crosstab(index, columns, values=values, aggfunc='sum')
    expected = pd.DataFrame({'one': [3, 1], 'two': [2, 4]}, index=['bar', 'foo'])
    assert_frame_equal(result, expected)

def test_crosstab_with_margins():
    index = ['foo', 'foo', 'bar', 'bar']
    columns = ['one', 'two', 'one', 'two']
    result = pd.crosstab(index, columns, margins=True)
    expected = pd.DataFrame({
        'one': [1, 2, 3],
        'two': [2, 2, 4],
        'All': [3, 4, 7]
    }, index=['bar', 'foo', 'All'])
    assert_frame_equal(result, expected)

def test_crosstab_with_normalize():
    index = ['foo', 'foo', 'bar', 'bar']
    columns = ['one', 'two', 'one', 'two']
    result = pd.crosstab(index, columns, normalize=True)
    expected = pd.DataFrame({
        'one': [1/3, 2/3],
        'two': [2/4, 2/4]
    }, index=['bar', 'foo'])
    assert_frame_equal(result, expected)

def test_crosstab_with_dropna_false():
    index = pd.Categorical(['foo', 'foo', 'bar'], categories=['foo', 'bar', 'baz'])
    columns = pd.Categorical(['one', 'two', 'three'], categories=['one', 'two', 'three', 'four'])
    result = pd.crosstab(index, columns, dropna=False)
    expected = pd.DataFrame({
        'one': [0, 1, 0],
        'two': [1, 0, 0],
        'three': [0, 0, 1],
        'four': [0, 0, 0]
    }, index=['bar', 'foo', 'baz'])
    assert_frame_equal(result, expected)

def test_crosstab_with_multiple_columns():
    index = ['foo', 'foo', 'bar', 'bar']
    columns = [['one', 'one', 'two', 'two'], ['A', 'B', 'A', 'B']]
    result = pd.crosstab(index, columns)
    expected = pd.DataFrame({
        ('one', 'A'): [0, 1],
        ('one', 'B'): [1, 0],
        ('two', 'A'): [1, 0],
        ('two', 'B'): [0, 1]
    }, index=['bar', 'foo'])
    assert_frame_equal(result, expected)

def test_crosstab_empty_input():
    index = []
    columns = []
    result = pd.crosstab(index, columns)
    expected = pd.DataFrame()
    assert_frame_equal(result, expected)

def test_crosstab_with_non_string_index():
    index = [1, 1, 2, 2]
    columns = ['A', 'B', 'A', 'B']
    result = pd.crosstab(index, columns)
    expected = pd.DataFrame({'A': [1, 1], 'B': [1, 1]}, index=[1, 2])
    assert_frame_equal(result, expected)

def test_crosstab_with_mixed_dtypes():
    index = [1, 1, 2, 2]
    columns = ['A', 'B', 'A', 'B']
    values = [1.1, 2.2, 3.3, 4.4]
    result = pd.crosstab(index, columns, values=values, aggfunc='sum')
    expected = pd.DataFrame({'A': [3.3, 1.1], 'B': [2.2, 4.4]}, index=[1, 2])
    assert_frame_equal(result, expected)

def test_crosstab_with_rownames_and_colnames():
    index = ['foo', 'foo', 'bar', 'bar']
    columns = ['one', 'two', 'one', 'two']
    result = pd.crosstab(index, columns, rownames=['row'], colnames=['col'])
    expected = pd.DataFrame({'col': {'one': [1, 2], 'two': [2, 2]}}, index=['bar', 'foo'])
    expected = expected.rename_axis(index='row')
    assert_frame_equal(result, expected)

def test_crosstab_with_invalid_input():
    index = ['foo', 'foo', 'bar', 'bar']
    columns = ['one', 'two', 'one', 'two']
    with pytest.raises(ValueError):
        pd.crosstab(index, columns, values=[1, 2, 3, 4])
    with pytest.raises(ValueError):
        pd.crosstab(index, columns, aggfunc='sum')