import pandas as pd
import pandas.testing as pd_testing
import pytest

def test_pivot_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        pd.pivot(df, columns='A', index='B')

def test_pivot_empty_series():
    s = pd.Series()
    with pytest.raises(ValueError):
        pd.pivot(s.to_frame(), columns='A', index='B')

def test_pivot_single_column():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({1: [None, None, 1], 2: [None, 2, None], 3: [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_single_index():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='B', index='A')
    expected = pd.DataFrame({4: [1], 5: [2], 6: [3]}, index=[1, 2, 3])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_multi_column():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = pd.pivot(df, columns='A', index='B', values='C')
    expected = pd.DataFrame({1: [7], 2: [8], 3: [9]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_multi_index():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = pd.pivot(df, columns='A', index=['B', 'C'])
    expected = pd.DataFrame({1: [None, None, 1], 2: [None, 2, None], 3: [3, None, None]}, index=pd.MultiIndex.from_tuples([(4, 7), (5, 8), (6, 9)], names=['B', 'C']))
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_axis_0():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B', axis=0)
    expected = pd.DataFrame({1: [None, None, 1], 2: [None, 2, None], 3: [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_axis_1():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='B', index='A', axis=1)
    expected = pd.DataFrame({4: [1], 5: [2], 6: [3]}, index=[1, 2, 3])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_join_outer():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = pd.pivot(df, columns='A', index='B', values='C', join='outer')
    expected = pd.DataFrame({1: [7], 2: [8], 3: [9]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_join_inner():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = pd.pivot(df, columns='A', index='B', values='C', join='inner')
    expected = pd.DataFrame({1: [7], 2: [8], 3: [9]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_sort():
    df = pd.DataFrame({'A': [3, 2, 1], 'B': [6, 5, 4], 'C': [9, 8, 7]})
    result = pd.pivot(df, columns='A', index='B', values='C', sort=True)
    expected = pd.DataFrame({1: [7], 2: [8], 3: [9]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_verify_integrity():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result = pd.pivot(df, columns='A', index='B', values='C', verify_integrity=True)
    expected = pd.DataFrame({1: [7], 2: [8], 3: [9]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_series_concat():
    s = pd.Series([1, 2, 3], index=[4, 5, 6])
    df = pd.DataFrame({'A': s})
    result = pd.pivot(df, columns='A', index='A')
    expected = pd.DataFrame({1: [None, None, 1], 2: [None, 2, None], 3: [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_dataframe_concat():
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    df = pd.concat([df1, df2])
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({1: [None, None, 1], 2: [None, 2, None], 3: [3, None, None], 7: [None, None, 7], 8: [None, 8, None], 9: [9, None, None]}, index=[4, 5, 6, 10, 11, 12])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_empty_index():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df.index = pd.Index([None, None, None])
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({1: [None, None, 1], 2: [None, 2, None], 3: [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_non_unique_index():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df.index = pd.Index([1, 1, 2])
    with pytest.raises(ValueError):
        pd.pivot(df, columns='A', index='B')

def test_pivot_multiindex():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    df.index = pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)], names=['X', 'Y'])
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({1: [None, None, 1], 2: [None, 2, None], 3: [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_datetime():
    df = pd.DataFrame({'A': pd.date_range('2022-01-01', periods=3), 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({pd.Timestamp('2022-01-01'): [None, None, 1], pd.Timestamp('2022-01-02'): [None, 2, None], pd.Timestamp('2022-01-03'): [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_categorical():
    df = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'c']), 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({'a': [None, None, 1], 'b': [None, 2, None], 'c': [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_bool():
    df = pd.DataFrame({'A': [True, False, True], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({True: [1, None, 3], False: [None, 2, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_object():
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({'a': [None, None, 1], 'b': [None, 2, None], 'c': [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_numeric():
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({1.0: [None, None, 1], 2.0: [None, 2, None], 3.0: [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)

def test_pivot_mixed_dtypes():
    df = pd.DataFrame({'A': [1, 'a', 3.0], 'B': [4, 5, 6]})
    result = pd.pivot(df, columns='A', index='B')
    expected = pd.DataFrame({1: [None, None, 1], 'a': [None, 2, None], 3.0: [3, None, None]}, index=[4, 5, 6])
    pd_testing.assert_frame_equal(result, expected)