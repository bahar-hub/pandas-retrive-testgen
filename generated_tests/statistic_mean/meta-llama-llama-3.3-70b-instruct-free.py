import pytest
from statistics import mean, StatisticsError

def test_mean_empty_list():
    with pytest.raises(StatisticsError):
        mean([])

def test_mean_single_element_list():
    assert mean([1]) == 1

def test_mean_multiple_elements_list():
    assert mean([1, 2, 3, 4, 4]) == 2.8

def test_mean_fraction_elements_list():
    from fractions import Fraction as F
    assert mean([F(3, 7), F(1, 21), F(5, 3), F(1, 3)]) == F(13, 21)

def test_mean_decimal_elements_list():
    from decimal import Decimal as D
    assert mean([D("0.5"), D("0.75"), D("0.625"), D("0.375")]) == D('0.5625')

def test_mean_negative_numbers_list():
    assert mean([-1, -2, -3, -4, -4]) == -2.8

def test_mean_mixed_numbers_list():
    assert mean([-1, 2, -3, 4, -4]) == -0.4

def test_mean_float_numbers_list():
    assert mean([1.1, 2.2, 3.3, 4.4, 4.4]) == 3.08

def test_mean_large_numbers_list():
    assert mean([1000000, 2000000, 3000000, 4000000, 4000000]) == 2800000

def test_mean_zero_list():
    assert mean([0, 0, 0, 0, 0]) == 0

def test_mean_iterator():
    assert mean(iter([1, 2, 3, 4, 4])) == 2.8

def test_mean_tuple():
    assert mean((1, 2, 3, 4, 4)) == 2.8

def test_mean_set():
    assert mean({1, 2, 3, 4, 4}) == 2.8

def test_mean_string():
    with pytest.raises(TypeError):
        mean("hello")

def test_mean_none():
    with pytest.raises(TypeError):
        mean([None, None, None, None, None])

def test_mean_mixed_types():
    with pytest.raises(TypeError):
        mean([1, "hello", 3, 4, 4])