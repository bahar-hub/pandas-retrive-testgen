import pytest
import pandas as pd
import numpy as np

def test_crosstab_empty_inputs():
    with pytest.raises(ValueError):
        pd.crosstab([], [])

def test_crosstab_single_input():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    result = pd.crosstab(a, a)
    expected = pd.DataFrame({"foo": [11], "bar": [0]}, index=["foo"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b)
    expected = pd.DataFrame({"one": [6, 4], "two": [5, 0]}, index=["bar", "foo"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_rownames():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b, rownames=["a"])
    expected = pd.DataFrame({"one": [6, 4], "two": [5, 0]}, index=pd.Index(["bar", "foo"], name="a"))
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_colnames():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b, colnames=["b"])
    expected = pd.DataFrame({"b": {"one": [6, 4], "two": [5, 0]}}, index=["bar", "foo"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_rownames_and_colnames():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b, rownames=["a"], colnames=["b"])
    expected = pd.DataFrame({"b": {"one": [6, 4], "two": [5, 0]}}, index=pd.Index(["bar", "foo"], name="a"))
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_margins():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b, margins=True)
    expected = pd.DataFrame({"one": [6, 4, 10], "two": [5, 0, 5], "All": [11, 11, 22]}, index=["bar", "foo", "All"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_margins_name():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b, margins=True, margins_name="Total")
    expected = pd.DataFrame({"one": [6, 4, 10], "two": [5, 0, 5], "Total": [11, 11, 22]}, index=["bar", "foo", "Total"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_dropna():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b, dropna=False)
    expected = pd.DataFrame({"one": [6, 4], "two": [5, 0]}, index=["bar", "foo"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_normalize():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    result = pd.crosstab(a, b, normalize=True)
    expected = pd.DataFrame({"one": [6/11, 4/11], "two": [5/11, 0/11]}, index=["bar", "foo"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_two_inputs_with_values():
    a = np.array(["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"], dtype=object)
    b = np.array(["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"], dtype=object)
    c = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int)
    result = pd.crosstab(a, b, values=c, aggfunc=np.sum)
    expected = pd.DataFrame({"one": [21, 15], "two": [36, 0]}, index=["bar", "foo"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_categorical():
    foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
    result = pd.crosstab(foo, bar)
    expected = pd.DataFrame({"d": [1, 0], "e": [0, 1]}, index=["a", "b"])
    pd.testing.assert_frame_equal(result, expected)

def test_crosstab_categorical_with_dropna():
    foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
    result = pd.crosstab(foo, bar, dropna=False)
    expected = pd.DataFrame({"d": [1, 0, 0], "e": [0, 1, 0], "f": [0, 0, 0]}, index=["a", "b", "c"])
    pd.testing.assert_frame_equal(result, expected)