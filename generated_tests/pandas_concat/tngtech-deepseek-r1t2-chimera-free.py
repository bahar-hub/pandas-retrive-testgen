import pytest
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandas.testing import assert_frame_equal, assert_series_equal
from pandas.api.types import is_categorical_dtype

def test_basic_series_concat():
    s1 = Series(['a', 'b'])
    s2 = Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = Series(['a', 'b', 'c', 'd'], index=[0,1,0,1])
    assert_series_equal(result, expected)

def test_dataframe_vertical_concat():
    df1 = DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = DataFrame({'A': [5, 6], 'B': [7, 8]})
    result = pd.concat([df1, df2])
    expected = DataFrame({'A': [1,2,5,6], 'B': [3,4,7,8]}, index=[0,1,0,1])
    assert_frame_equal(result, expected)

def test_horizontal_concat():
    df1 = DataFrame({'A': [1, 2]})
    df2 = DataFrame({'B': [3, 4]})
    result = pd.concat([df1, df2], axis=1)
    expected = DataFrame({'A': [1,2], 'B': [3,4]})
    assert_frame_equal(result, expected)

def test_ignore_index():
    df1 = DataFrame({'A': [1, 2]}, index=[1, 2])
    df2 = DataFrame({'A': [3, 4]}, index=[3, 4])
    result = pd.concat([df1, df2], ignore_index=True)
    expected = DataFrame({'A': [1,2,3,4]}, index=[0,1,2,3])
    assert_frame_equal(result, expected)

def test_keys_with_series():
    s1 = Series([1, 2])
    s2 = Series([3, 4])
    result = pd.concat([s1, s2], keys=['a', 'b'])
    expected_index = pd.MultiIndex.from_tuples([('a',0),('a',1),('b',0),('b',1)])
    expected = Series([1,2,3,4], index=expected_index)
    assert_series_equal(result, expected)

def test_names_with_keys():
    df1 = DataFrame({'A': [1, 2]})
    df2 = DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], keys=['x', 'y'], names=['group', 'id'])
    expected_index = pd.MultiIndex.from_tuples(
        [('x',0),('x',1),('y',0),('y',1)], names=['group','id']
    )
    expected = DataFrame({'A': [1,2,3,4]}, index=expected_index)
    assert_frame_equal(result, expected)

def test_inner_join():
    df1 = DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2], join='inner')
    expected = DataFrame({'A': [1,2,5,6]}, index=[0,1,0,1])
    assert_frame_equal(result, expected)

def test_outer_join():
    df1 = DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2], join='outer')
    expected = DataFrame({
        'A': [1,2,5,6],
        'B': [3,4,np.nan,np.nan],
        'C': [np.nan,np.nan,7,8]
    }, index=[0,1,0,1])
    assert_frame_equal(result, expected)

def test_sort_non_concat_axis():
    df1 = DataFrame({'B': [1, 2], 'A': [3, 4]}, columns=['B','A'])
    df2 = DataFrame({'C': [5, 6], 'A': [7, 8]}, columns=['C','A'])
    result = pd.concat([df1, df2], sort=True)
    expected = DataFrame({
        'A': [3,4,7,8],
        'B': [1,2,np.nan,np.nan],
        'C': [np.nan,np.nan,5,6]
    }, columns=['A','B','C'], index=[0,1,0,1])
    assert_frame_equal(result, expected)

def test_verify_integrity():
    df1 = DataFrame({'A': [1]}, index=['x'])
    df2 = DataFrame({'A': [2]}, index=['x'])
    with pytest.raises(ValueError):
        pd.concat([df1, df2], verify_integrity=True)

def test_empty_objects():
    df1 = DataFrame()
    df2 = DataFrame({'A': [1, 2]})
    result = pd.concat([df1, df2])
    assert_frame_equal(result, df2)

def test_mixed_series_dataframe():
    s = Series([1, 2], name='A')
    df = DataFrame({'B': [3, 4]})
    result = pd.concat([s, df], axis=1)
    expected = DataFrame({'A': [1,2], 'B': [3,4]})
    assert_frame_equal(result, expected)

def test_non_unique_columns():
    df1 = DataFrame([[1, 2]], columns=['A', 'A'])
    df2 = DataFrame([[3, 4]], columns=['B', 'B'])
    result = pd.concat([df1, df2], axis=1)
    expected = DataFrame([[1,2,3,4]], columns=['A','A','B','B'])
    assert_frame_equal(result, expected)

def test_multiindex_rows():
    index = pd.MultiIndex.from_tuples([('a',1), ('a',2)])
    df1 = DataFrame({'A': [1, 2]}, index=index)
    df2 = DataFrame({'A': [3, 4]}, index=index)
    result = pd.concat([df1, df2])
    expected_index = pd.MultiIndex.from_tuples(
        [('a',1),('a',2),('a',1),('a',2)], names=[None, None]
    )
    expected = DataFrame({'A': [1,2,3,4]}, index=expected_index)
    assert_frame_equal(result, expected)

def test_categorical_preservation():
    df1 = DataFrame({'A': pd.Categorical(['a', 'b'])})
    df2 = DataFrame({'A': pd.Categorical(['c', 'd'])})
    result = pd.concat([df1, df2])
    assert is_categorical_dtype(result['A'])
    expected = DataFrame({'A': ['a','b','c','d']}, dtype='category')
    assert_frame_equal(result, expected)

def test_datetime_with_timezone():
    dt = pd.date_range('2020-01-01', periods=2, tz='UTC')
    df1 = DataFrame({'A': dt})
    df2 = DataFrame({'A': dt + pd.Timedelta(days=1)})
    result = pd.concat([df1, df2])
    expected = DataFrame({'A': list(dt) + list(dt + pd.Timedelta(days=1))})
    assert_frame_equal(result, expected)

def test_boolean_object_mix():
    df1 = DataFrame({'A': [True, False]})
    df2 = DataFrame({'A': ['x', 'y']})
    result = pd.concat([df1, df2])
    expected = DataFrame({'A': [True, False, 'x', 'y']})
    assert_frame_equal(result, expected)

def test_copy_observable():
    df = DataFrame({'A': [1, 2]})
    df2 = DataFrame({'B': [3, 4]})
    result = pd.concat([df, df2], copy=False)
    assert result._mgr.blocks[0].values.base is None  # No base means it's a copy

def test_levels_parameter():
    df1 = DataFrame({'A': [1, 2]})
    df2 = DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2], keys=['x', 'y'], levels=[['x','y']])
    assert result.index.levels[0].tolist() == ['x','y']

def test_all_none_objects():
    with pytest.raises(ValueError):
        pd.concat([None, None])

def test_mapping_input():
    df1 = DataFrame({'A': [1, 2]})
    df2 = DataFrame({'A': [3, 4]})
    result = pd.concat({'x': df1, 'y': df2})
    expected_index = pd.MultiIndex.from_tuples(
        [('x',0),('x',1),('y',0),('y',1)], names=[None, None]
    )
    expected = DataFrame({'A': [1,2,3,4]}, index=expected_index)
    assert_frame_equal(result, expected)