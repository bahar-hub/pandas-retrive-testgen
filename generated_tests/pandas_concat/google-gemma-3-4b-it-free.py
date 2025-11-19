import pandas as pd
import pandas.testing as t

def test_concat_series():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    t.assert_series_equal(pd.concat([s1, s2]), pd.Series(['a', 'b', 'c', 'd']))

def test_concat_series_ignore_index():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    t.assert_series_equal(pd.concat([s1, s2], ignore_index=True), pd.Series(['a', 'b', 'c', 'd']))

def test_concat_series_keys():
    s1 = pd.Series(['a', 'b'])
    s2 = pd.Series(['c', 'd'])
    result = pd.concat([s1, s2], keys=['s1', 's2'])
    t.assert_series_equal(result['s1'], s1)
    t.assert_series_equal(result['s2'], s2)

def test_concat_dataframe():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    t.assert_frame_equal(pd.concat([df1, df2]),
                         pd.DataFrame({'A': [1, 5, 2, 6], 'B': [3, 4, 7, 8], 'C': [None, None, None, None]})
                         )

def test_concat_dataframe_ignore_index():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    t.assert_frame_equal(pd.concat([df1, df2], ignore_index=True),
                         pd.DataFrame({'A': [1, 5, 2, 6], 'B': [3, 4, 7, 8], 'C': [None, None, None, None]})
                         )

def test_concat_dataframe_keys():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2], keys=['df1', 'df2'])
    t.assert_frame_equal(result.loc['df1'], df1)
    t.assert_frame_equal(result.loc['df2'], df2)

def test_concat_mixed_dtypes():
    s1 = pd.Series([1, 2, 3], dtype='int64')
    s2 = pd.Series(['a', 'b', 'c'], dtype='object')
    result = pd.concat([s1, s2])
    t.assert_series_equal(result, pd.Series(['1', '2', '3', 'a', 'b', 'c'], dtype='object'))

def test_concat_empty_series():
    s1 = pd.Series([])
    s2 = pd.Series([])
    t.assert_series_equal(pd.concat([s1, s2]), pd.Series([]))

def test_concat_empty_dataframe():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    t.assert_frame_equal(pd.concat([df1, df2]), pd.DataFrame())

def test_concat_different_column_sets():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': [1, 5, 2, 6], 'B': [3, 4, None, None], 'C': [None, 7, None, 8]}))

def test_concat_categorical_dtype():
    df1 = pd.DataFrame({'A': pd.Categorical(['a', 'b'])})
    df2 = pd.DataFrame({'A': pd.Categorical(['c', 'd'])})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': pd.Categorical(['a', 'b', 'c', 'd'])}))

def test_concat_datetime_dtype():
    df1 = pd.DataFrame({'A': pd.to_datetime(['2023-01-01', '2023-01-02'])})
    df2 = pd.DataFrame({'A': pd.to_datetime(['2023-01-03', '2023-01-04'])})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])}))

def test_concat_boolean_dtype():
    df1 = pd.DataFrame({'A': [True, False]})
    df2 = pd.DataFrame({'A': [False, True]})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': [True, False, False, True]}))

def test_concat_object_dtype():
    df1 = pd.DataFrame({'A': ['a', 'b']})
    df2 = pd.DataFrame({'A': ['c', 'd']})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': ['a', 'b', 'c', 'd']}))

def test_concat_numeric_dtype():
    df1 = pd.DataFrame({'A': [1, 2]})
    df2 = pd.DataFrame({'A': [3, 4]})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': [1, 2, 3, 4]}))

def test_concat_non_unique_indices():
    df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 0])
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': [1, 2, 3, 4]}, index=[0, 1]))

def test_concat_non_unique_columns():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    result = pd.concat([df1, df2], join='outer')
    t.assert_frame_equal(result, pd.DataFrame({'A': [1, 5, 2, 6], 'B': [3, 7, 4, 8]}))

def test_concat_multiindex_rows():
    index1 = pd.MultiIndex.from_tuples([('a', 1), ('b', 2)], names=['level_1', 'level_2'])
    index2 = pd.MultiIndex.from_tuples([('a', 3), ('c', 4)], names=['level_1', 'level_2'])
    df1 = pd.DataFrame({'A': [1, 2]}, index=index1)
    df2 = pd.DataFrame({'A': [3, 4]}, index=index2)
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': [1, 2, 3, 4]}, index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2), ('a', 3), ('c', 4)])))

def test_concat_multiindex_columns():
    columns1 = pd.MultiIndex.from_tuples([('x', 1), ('y', 2)], names=['level_1', 'level_2'])
    columns2 = pd.MultiIndex.from_tuples([('x', 3), ('z', 4)], names=['level_1', 'level_2'])
    df1 = pd.DataFrame({'A': [1, 2]}, columns=columns1)
    df2 = pd.DataFrame({'A': [3, 4]}, columns=columns2)
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': [1, 2, 3, 4]}, columns=pd.MultiIndex.from_tuples([('x', 1), ('y', 2), ('x', 3), ('z', 4)])))

def test_concat_differing_column_sets():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6], 'D': [7, 8]}))

def test_concat_categorical_and_numeric():
    df1 = pd.DataFrame({'A': pd.Categorical(['a', 'b']), 'B': [1, 2]})
    df2 = pd.DataFrame({'A': pd.Categorical(['c', 'd']), 'B': [3, 4]})
    result = pd.concat([df1, df2])
    t.assert_frame_equal(result, pd.DataFrame({'A': pd.Categorical(['a', 'b', 'c', 'd']), 'B': [1, 2, 3, 4]}))