import datetime
import pytest
import pandas as pd
from pandas import Series, DataFrame, DatetimeIndex, NaT, Timestamp
from pandas.testing import assert_series_equal, assert_index_equal

def test_scalar_integer():
    result = pd.to_datetime(1490195805, unit='s')
    assert result == Timestamp('2017-03-22 15:16:45')

def test_scalar_float():
    result = pd.to_datetime(1490195805.0, unit='s')
    assert result == Timestamp('2017-03-22 15:16:45')

def test_scalar_string():
    result = pd.to_datetime('2023-01-01')
    assert result == Timestamp('2023-01-01')

def test_scalar_datetime():
    dt = datetime.datetime(2023, 1, 1)
    result = pd.to_datetime(dt)
    assert result == Timestamp(dt)

def test_list_strings():
    result = pd.to_datetime(['2023-01-01', '2023-01-02'])
    expected = DatetimeIndex(['2023-01-01', '2023-01-02'])
    assert_index_equal(result, expected)

def test_list_integers_with_unit():
    result = pd.to_datetime([1, 2, 3], unit='D', origin='2023-01-01')
    expected = DatetimeIndex(['2023-01-02', '2023-01-03', '2023-01-04'])
    assert_index_equal(result, expected)

def test_series_strings():
    s = Series(['2023-01-01', '2023-01-02'])
    result = pd.to_datetime(s)
    expected = Series(DatetimeIndex(['2023-01-01', '2023-01-02']))
    assert_series_equal(result, expected)

def test_series_integers_with_unit():
    s = Series([1, 2, 3])
    result = pd.to_datetime(s, unit='D', origin='2023-01-01')
    expected = Series(DatetimeIndex(['2023-01-02', '2023-01-03', '2023-01-04']))
    assert_series_equal(result, expected)

def test_dataframe_valid():
    df = DataFrame({'year': [2023, 2024], 'month': [1, 2], 'day': [1, 1]})
    result = pd.to_datetime(df)
    expected = Series(DatetimeIndex(['2023-01-01', '2024-02-01']))
    assert_series_equal(result, expected)

def test_dataframe_invalid_missing_column():
    df = DataFrame({'year': [2023], 'month': [1]})
    with pytest.raises(ValueError):
        pd.to_datetime(df)

def test_errors_coerce():
    result = pd.to_datetime('invalid', errors='coerce')
    assert result is NaT

def test_errors_ignore():
    result = pd.to_datetime('invalid', errors='ignore')
    assert result == 'invalid'

def test_errors_raise():
    with pytest.raises(ValueError):
        pd.to_datetime('invalid', errors='raise')

def test_dayfirst():
    result = pd.to_datetime('01-02-2023', dayfirst=True)
    assert result == Timestamp('2023-02-01')

def test_yearfirst():
    result = pd.to_datetime('01-02-2023', yearfirst=True)
    assert result == Timestamp('2023-01-02')

def test_utc_localize():
    result = pd.to_datetime('2023-01-01', utc=True)
    assert result == Timestamp('2023-01-01', tz='UTC')

def test_format_specified():
    result = pd.to_datetime('20230101', format='%Y%m%d')
    assert result == Timestamp('2023-01-01')

def test_unit_ns():
    result = pd.to_datetime(1490195805433502912, unit='ns')
    assert result == Timestamp('2017-03-22 15:16:45.433502912')

def test_origin_julian():
    result = pd.to_datetime(1, unit='D', origin='julian')
    expected = Timestamp('4713-01-02 12:00:00')
    assert result == expected

def test_empty_list():
    result = pd.to_datetime([], errors='coerce')
    assert_index_equal(result, DatetimeIndex([]))

def test_none_input():
    assert pd.to_datetime(None) is NaT

def test_mixed_timezones_with_utc():
    # Should convert to UTC without warning
    result = pd.to_datetime(['2020-01-01 01:00:00-01:00', '2020-01-01 02:00:00+01:00'], utc=True)
    expected = DatetimeIndex(['2020-01-01 02:00:00+00:00', '2020-01-01 01:00:00+00:00'], tz='UTC')
    assert_index_equal(result, expected)

def test_out_of_bounds():
    with pytest.raises(ValueError):
        pd.to_datetime('1300-01-01', errors='raise')

def test_out_of_bounds_coerce():
    result = pd.to_datetime('1300-01-01', errors='coerce')
    assert result is NaT

def test_mixed_string_and_datetime():
    dt = datetime.datetime(2023, 1, 1)
    result = pd.to_datetime(['2023-01-01', dt], utc=True)
    expected = DatetimeIndex(['2023-01-01 00:00:00+00:00', '2023-01-01 00:00:00+00:00'], tz='UTC')
    assert_index_equal(result, expected)

def test_exact_format_matching():
    result = pd.to_datetime('2023-01-01 12:00', format='%Y-%m-%d %H:%M', exact=True)
    assert result == Timestamp('2023-01-01 12:00')

def test_cache_duplicate_strings():
    # Cache behavior is internal, but we can test duplicate handling
    dates = ['2023-01-01'] * 100
    result = pd.to_datetime(dates)
    expected = DatetimeIndex([Timestamp('2023-01-01')] * 100)
    assert_index_equal(result, expected)