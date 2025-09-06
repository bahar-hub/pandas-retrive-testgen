import pytest
import pandas as pd
import numpy as np

def test_concat_simple_series():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2])
    expected = pd.Series(['a', 'b', 'c', 'd'], name=0)
    pd.testing.assert_series_equal(result, expected)

def test_concat_simple_dataframe():
    df1 = pd.DataFrame({'letter': ['a', 'b'], 'number': [1, 2]})
    df2 = pd.DataFrame({'letter': ['c', 'd'], 'number': [3, 4]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({
        'letter': ['a', 'b', 'c', 'd'],
        'number': [1, 2, 3, 4]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_concat_with_ignore_index():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], ignore_index=True)
    expected = pd.Series(['a', 'b', 'c', 'd'], name=0)
    pd.testing.assert_series_equal(result, expected)

def test_concat_with_axis_1():
    df1 = pd.DataFrame({'letter': ['a', 'b']})
    df2 = pd.DataFrame({'number': [1, 2]})
    result = pd.concat([df1, df2], axis=1)
    expected = pd.DataFrame({
        'letter': ['a', 'b'],
        'number': [1, 2]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_concat_with_inner_join():
    df1 = pd.DataFrame({'letter': ['a', 'b'], 'number': [1, 2]})
    df3 = pd.DataFrame({'letter': ['c', 'd'], 'number': [3, 4], 'animal': ['cat', 'dog']})
    result = pd.concat([df1, df3], join='inner')
    expected = pd.DataFrame({
        'letter': ['a', 'b'],
        'number': [1, 2]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_concat_with_outer_join():
    df1 = pd.DataFrame({'letter': ['a', 'b'], 'number': [1, 2]})
    df3 = pd.DataFrame({'letter': ['c', 'd'], 'number': [3, 4], 'animal': ['cat', 'dog']})
    result = pd.concat([df1, df3], join='outer')
    expected = pd.DataFrame({
        'letter': ['a', 'b', 'c', 'd'],
        'number': [1, 2, 3, 4],
        'animal': [np.nan, np.nan, 'cat', 'dog']
    })
    pd.testing.assert_frame_equal(result, expected)

def test_concat_with_verify_integrity():
    df5 = pd.DataFrame([1], index=['a'])
    df6 = pd.DataFrame([2], index=['a'])
    with pytest.raises(ValueError):
        pd.concat([df5, df6], verify_integrity=True)

def test_concat_with_keys():
    s1 = pd.Series(['a', 'b'], name='s1')
    s2 = pd.Series(['c', 'd'], name='s2')
    result = pd.concat([s1, s2], keys=['key1', 'key2'])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=pd.MultiIndex.from_tuples([('key1', 0), ('key1', 1), ('key2', 0), ('key2', 1)], names=('keys', None)), name='s1')
    pd.testing.assert_series_equal(result, expected)

def test_concat_with_names():
    s1 = pd.Series(['a', 'b'], name='s1')
    s2 = pd.Series(['c', 'd'], name='s2')
    result = pd.concat([s1, s2], keys=['key1', 'key2'], names=['level1', 'level2'])
    expected = pd.Series(['a', 'b', 'c', 'd'], index=pd.MultiIndex.from_tuples([('key1', 0), ('key1', 1), ('key2', 0), ('key2', 1)], names=('level1', 'level2')), name='s1')
    pd.testing.assert_series_equal(result, expected)

def test_concat_with_multiindex():
    index = pd.MultiIndex.from_tuples([('x', 1), ('x', 2)], names=['letter', 'number'])
    df1 = pd.DataFrame({'value': [10, 20]}, index=index)
    result = pd.concat([df1])
    expected = pd.DataFrame({'value': [10, 20]}, index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)], names=['letter', 'number']))
    pd.testing.assert_frame_equal(result, expected)

def test_concat_with_non_unique_indices():
    df1 = pd.DataFrame({'letter': ['a', 'b'], 'number': [1, 2]})
    df2 = pd.DataFrame({'letter': ['a', 'b'], 'number': [3, 4]})
    result = pd.concat([df1, df2])
    expected = pd.DataFrame({
        'letter': ['a', 'b', 'a', 'b'],
        'number': [1, 2, 3, 4]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_concat_with_mixed_dtypes():
    s1 = pd.Series([1, 2], dtype=int)
    s2 = pd.Series(['a', 'b'], dtype=object)
    result = pd.concat([s1, s2])
    expected = pd.Series([1, 2, np.nan, np.nan], dtype=float)
    pd.testing.assert_series_equal(result, expected)

def test_concat_with_categorical_dtypes():
    s1 = pd.Series(pd.Categorical(['a', 'b']))
    s2 = pd.Series(pd.Categorical(['c', 'd']))
    result = pd.concat([s1, s2])
    expected = pd.Series(pd.Categorical(['a', 'b', 'c', 'd']))
    pd.testing.assert_series_equal(result, expected)

def test_concat_with_datetime_dtypes():
    s1 = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02']), dtype='datetime64[ns]')
    s2 = pd.Series(pd.to_datetime(['2023-01-03', '2023-01-04'], tz='US/Eastern'), dtype='datetime64[ns, US/Eastern]')
    result = pd.concat([s1, s2])
    expected = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'], tz='US/Eastern'), dtype='datetime64[ns, US/Eastern]')
    pd.testing.assert_series_equal(result, expected)