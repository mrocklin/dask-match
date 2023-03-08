import os

import pandas as pd
import numpy as np
import pytest
from dask.dataframe.utils import assert_eq
from dask.utils import M

from dask_match import ReadCSV, from_pandas, optimize, read_parquet


def _make_file(dir, format="parquet", df=None): 
    fn = os.path.join(str(dir), f"myfile.{format}")
    if df is None:
        df = pd.DataFrame({c: range(10) for c in "abcde"})
    if format == "csv":
        df.to_csv(fn)
    elif format == "parquet":
        df.to_parquet(fn)
    else:
        ValueError(f"{format} not a supported format")
    return fn


def test_basic(tmpdir):
    fn_pq = _make_file(tmpdir, format="parquet")
    fn_csv = _make_file(tmpdir, format="csv")

    x = read_parquet(fn_pq, columns=("a", "b", "c"))
    y = ReadCSV(fn_csv, usecols=("a", "d", "e"))

    z = x + y
    result = z[("a", "b", "d")].sum(skipna="foo")
    assert result.operand("skipna") == "foo"
    assert result.operands[0].columns == ("a", "b", "d")

    x + 1
    1 + x


def df(fn):
    return read_parquet(fn, columns=["a", "b", "c"])


def df_bc(fn):
    return read_parquet(fn, columns=["b", "c"])


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            # Add -> Mul
            lambda fn: df(fn) + df(fn),
            lambda fn: 2 * df(fn),
        ),
        (
            # Column projection
            lambda fn: df(fn)[["b", "c"]],
            lambda fn: read_parquet(fn, columns=["b", "c"]),
        ),
        (
            # Compound
            lambda fn: 3 * (df(fn) + df(fn))[["b", "c"]],
            lambda fn: 6 * df_bc(fn),
        ),
        (
            # Traverse Sum
            lambda fn: df(fn).sum()[["b", "c"]],
            lambda fn: df_bc(fn).sum(),
        ),
        (
            # Respect Sum keywords
            lambda fn: df(fn).sum(numeric_only=True)[["b", "c"]],
            lambda fn: df_bc(fn).sum(numeric_only=True),
        ),
    ],
)
def test_optimize(tmpdir, input, expected):
    fn = _make_file(tmpdir, format="parquet")
    result = optimize(input(fn))
    assert str(result) == str(expected(fn))


def test_meta_divisions_name():
    a = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    df = 2 * from_pandas(a, npartitions=2)
    assert list(df.columns) == list(a.columns)
    assert df.npartitions == 2

    assert df.x.sum()._meta == 0
    assert df.x.sum().npartitions == 1

    assert "mul" in df._name
    assert "sum" in df.sum()._name


def test_meta_blockwise():
    a = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    b = pd.DataFrame({"z": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})

    aa = from_pandas(a, npartitions=2)
    bb = from_pandas(b, npartitions=2)

    cc = 2 * aa - 3 * bb
    assert set(cc.columns) == {"x", "y", "z"}


def test_dask():
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    df["y"] = df.y * 10.0

    ddf = from_pandas(df, npartitions=10)
    assert (ddf.x + ddf.y).npartitions == 10
    z = (ddf.x + ddf.y).sum()

    assert z.compute() == (df.x + df.y).sum()


@pytest.mark.parametrize(
    "func",
    [
        M.max,
        M.min,
        M.sum,
        M.count,
        pytest.param(
            M.mean,
            marks=pytest.mark.skip(reason="scalars don't work yet"),
        ),
        pytest.param(
            lambda df: df.size,
            marks=pytest.mark.skip(reason="scalars don't work yet"),
        ),
    ],
)
def test_reductions(func):
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    df["y"] = df.y * 10.0
    ddf = from_pandas(df, npartitions=10)

    assert_eq(func(ddf), func(df))
    assert func(ddf.x).compute() == func(df.x)


def test_mode():
    df = pd.DataFrame({"x": [1, 2, 3, 1, 2]})
    ddf = from_pandas(df, npartitions=3)

    assert_eq(ddf.x.mode(), df.x.mode(), check_names=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.x > 10,
        lambda df: df.x + 20 > df.y,
        lambda df: 10 < df.x,
        lambda df: 10 <= df.x,
        lambda df: 10 == df.x,
        lambda df: df.x < df.y,
        lambda df: df.x > df.y,
        lambda df: df.x == df.y,
        lambda df: df.x != df.y,
    ],
)
def test_conditionals(func):
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    df["y"] = df.y * 2.0
    ddf = from_pandas(df, npartitions=10)

    assert_eq(func(df), func(ddf), check_names=False)


def test_predicate_pushdown(tmpdir):
    from dask_match.io.parquet import ReadParquet

    original = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 10,
            "b": [0, 1, 2, 3, 4] * 10,
            "c": range(50),
            "d": [6, 7] * 25,
            "e": [8, 9] * 25,
        }
    )
    fn = _make_file(tmpdir, format="parquet", df=original)
    df = read_parquet(fn)
    assert_eq(df, original)
    x = df[df.a == 5][df.c > 20]["b"]
    y = optimize(x)
    assert isinstance(y, ReadParquet)
    assert ("a", "==", 5) in y.operand("filters") or ("a", "==", 5) in y.operand("filters")
    assert ("c", ">", 20) in y.operand("filters")
    assert y.columns == ["b"]

    # Check computed result
    y_result = y.compute()
    assert list(y_result.columns) == ["b"]
    assert len(y_result["b"]) == 6
    assert all(y_result["b"] == 4)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.astype(int),
        lambda df: df.apply(lambda row, x, y=10: row * x + y, x=2),
        lambda df: df[df.x > 5],
    ],
)
def test_blockwise(func):
    df = pd.DataFrame({"x": range(20), "y": range(20)})
    df["y"] = df.y * 2.0
    ddf = from_pandas(df, npartitions=3)

    assert_eq(func(df), func(ddf))


def test_repr():
    df = pd.DataFrame({"x": range(20), "y": range(20)})
    df = from_pandas(df, npartitions=1)

    assert "+ 1" in str(df + 1)
    assert "+ 1" in repr(df + 1)

    s = (df["x"] + 1).sum(skipna=False)
    assert '["x"]' in s or "['x']" in s
    assert "+ 1" in s
    assert "sum(skipna=False)" in s

    assert "ReadParquet" in read_parquet("filename")


def test_columns_traverse_filters():
    df = pd.DataFrame({"x": range(20), "y": range(20), "z": range(20)})
    df = from_pandas(df, npartitions=2)

    expr = df[df.x > 5].y
    result = optimize(expr)
    expected = df.y[df.x > 5]

    assert str(result) == str(expected)


def test_persist():
    df = pd.DataFrame({"x": range(20), "y": range(20), "z": range(20)})
    ddf = from_pandas(df, npartitions=2)

    a = ddf + 2
    b = a.persist()

    assert_eq(a, b)
    assert len(a.__dask_graph__()) > len(b.__dask_graph__())

    assert len(b.__dask_graph__()) == b.npartitions

    assert b.y.sum().compute() == (df + 2).y.sum()
