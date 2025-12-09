import pytest
from statistics import mean, StatisticsError
from fractions import Fraction
from decimal import Decimal

def test_mean_basic():
    assert mean([1, 2, 3, 4, 4]) == 2.8

def test_mean_fractions():
    assert mean([Fraction(3, 7), Fraction(1, 21), Fraction(5, 3), Fraction(1, 3)]) == Fraction(13, 21)

def test_mean_decimals():
    assert mean([Decimal("0.5"), Decimal("0.75"), Decimal("0.625"), Decimal("0.375")]) == Decimal('0.5625')

def test_mean_empty_list():
    with pytest.raises(StatisticsError):
        mean([])

def test_mean_single_element():
    assert mean([5]) == 5

def test_mean_negative_numbers():
    assert mean([-1, -2, -3, -4, -4]) == -2.8

def test_mean_mixed_numbers():
    assert mean([-1, 0, 1, 2, 3]) == 1.0

def test_mean_floats():
    assert mean([1.5, 2.5, 3.5, 4.5, 4.5]) == 3.2

def test_mean_large_numbers():
    assert mean([1000000, 2000000, 3000000, 4000000, 4000000]) == 2800000.0

def test_mean_iterable():
    class Iterable:
        def __iter__(self):
            return iter([1, 2, 3, 4, 4])
    assert mean(Iterable()) == 2.8

def test_mean_with_duplicates():
    assert mean([1, 1, 1, 1, 1]) == 1.0

def test_mean_with_zero():
    assert mean([0, 0, 0, 0, 0]) == 0.0

def test_mean_with_large_fraction():
    assert mean([Fraction(1000000, 1), Fraction(2000000, 1)]) == 1500000.0

def test_mean_with_large_decimal():
    assert mean([Decimal("1000000.5"), Decimal("2000000.5")]) == Decimal('1500000.5')

def test_mean_with_mixed_types():
    with pytest.raises(TypeError):
        mean([1, 2, "3", 4, 4])

def test_mean_with_non_numeric():
    with pytest.raises(TypeError):
        mean(["a", "b", "c"])