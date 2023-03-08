import functools
import math
import numbers
import operator
from collections.abc import Iterator

import toolz
from dask.base import DaskMethodsMixin, named_schedulers, normalize_token, tokenize
from dask.dataframe.core import _concat, is_dataframe_like
from dask.utils import M, apply, funcname
from matchpy import (
    Arity,
    CustomConstraint,
    Operation,
    Pattern,
    ReplacementRule,
    Wildcard,
    replace_all,
)
from matchpy.expressions.expressions import _OperationMeta

replacement_rules = []


class _ExprMeta(_OperationMeta):
    """Metaclass to determine Operation behavior

    Matchpy overrides `__call__` so that `__init__` doesn't behave as expected.
    This is gross, but has some logic behind it.  We need to enforce that
    expressions can be easily replayed.  Eliminating `__init__` is one way to
    do this.  It forces us to compute things lazily rather than at
    initialization.

    We motidify Matchpy's implementation so that we can handle keywords and
    default values more cleanly.

    We also collect replacement rules here.
    """

    seen = set()

    def __call__(cls, *args, variable_name=None, **kwargs):
        # Collect optimization rules for new classes
        if cls not in _ExprMeta.seen:
            _ExprMeta.seen.add(cls)
            for rule in cls._replacement_rules():
                replacement_rules.append(rule)

        # Grab keywords and manage default values
        operands = list(args)
        for parameter in cls._parameters[len(operands) :]:
            operands.append(kwargs.pop(parameter, cls._defaults[parameter]))
        assert not kwargs

        # Defer up to matchpy
        return super().__call__(*operands, variable_name=None)


_defer_to_matchpy = False


class Expr(Operation, DaskMethodsMixin, metaclass=_ExprMeta):
    """Primary class for all Expressions

    This mostly includes Dask protocols and various Pandas-like method
    definitions to make us look more like a DataFrame.
    """

    commutative = False
    associative = False
    _parameters = []
    _defaults = {}

    __dask_scheduler__ = staticmethod(
        named_schedulers.get("threads", named_schedulers["sync"])
    )
    __dask_optimize__ = staticmethod(lambda dsk, keys, **kwargs: dsk)

    @classmethod
    def _replacement_rules(cls) -> Iterator[ReplacementRule]:
        """Rules associated to this class that are useful for optimization

        See also:
            optimize
            _ExprMeta
        """
        yield from []

    def __str__(self):
        s = ", ".join(
            str(param) + "=" + str(operand)
            for param, operand in zip(self._parameters, self.operands)
            if operand != self._defaults.get(param)
        )
        return f"{type(self).__name__}({s})"

    def __repr__(self):
        return str(self)

    def __getattr__(self, key):
        if key == "__name__":
            return object.__getattribute__(self, key)
        elif key in dir(type(self)):
            return object.__getattribute__(self, key)
        elif is_dataframe_like(self._meta) and key in self._meta.columns:
            return self[key]
        else:
            return object.__getattribute__(self, key)

    def operand(self, key):
        return self.operands[type(self)._parameters.index(key)]

    @property
    def index(self):
        return Index(self)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    def __getitem__(self, other):
        if isinstance(other, Expr):
            return Filter(self, other)  # df[df.x > 1]
        else:
            return Projection(self, other)  # df[["a", "b", "c"]]

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __lt__(self, other):
        return LT(self, other)

    def __rlt__(self, other):
        return LT(other, self)

    def __gt__(self, other):
        return GT(self, other)

    def __rgt__(self, other):
        return GT(other, self)

    def __le__(self, other):
        return LE(self, other)

    def __rle__(self, other):
        return LE(other, self)

    def __ge__(self, other):
        return GE(self, other)

    def __rge__(self, other):
        return GE(other, self)

    def __eq__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__eq__(self, other)
        else:
            return EQ(other, self)

    def __ne__(self, other):
        if _defer_to_matchpy:  # Defer to matchpy when optimizing
            return Operation.__ne__(self, other)
        else:
            return NE(other, self)

    def sum(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return Sum(self, skipna, level, numeric_only, min_count)

    def mean(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return self.sum(skipna=skipna) / self.count()

    def max(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return Max(self, skipna, level, numeric_only, min_count)

    def mode(self, dropna=True):
        return Mode(self, dropna=dropna)

    def min(self, skipna=True, level=None, numeric_only=None, min_count=0):
        return Min(self, skipna, level, numeric_only, min_count)

    def count(self, numeric_only=None):
        return Count(self, numeric_only)

    @property
    def size(self):
        return Size(self)

    def astype(self, dtypes):
        return AsType(self, dtypes)

    def apply(self, function, *args, **kwargs):
        return Apply(self, function, args, kwargs)

    @property
    def divisions(self):
        if "divisions" in self._parameters:
            idx = self._parameters.index("divisions")
            return self.operands[idx]
        return tuple(self._divisions())

    @property
    def dask(self):
        return self.__dask_graph__()

    @property
    def known_divisions(self):
        """Whether divisions are already known"""
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self):
        if "npartitions" in self._parameters:
            idx = self._parameters.index("npartitions")
            return self.operands[idx]
        else:
            return len(self.divisions) - 1

    @property
    def _name(self):
        if "_name" in self._parameters:
            idx = self._parameters.index("_name")
            return self.operands[idx]
        return funcname(type(self)).lower() + "-" + tokenize(*self.operands)

    @property
    def columns(self):
        if "columns" in self._parameters:
            idx = self._parameters.index("columns")
            return self.operands[idx]
        else:
            return self._meta.columns

    @property
    def dtypes(self):
        return self._meta.dtypes

    @property
    def _meta(self):
        if "_meta" in self._parameters:
            idx = self._parameters.index("_meta")
            return self.operands[idx]
        raise NotImplementedError()

    def _divisions(self):
        raise NotImplementedError()

    def __dask_graph__(self):
        """Traverse expression tree, collect layers"""
        stack = [self]
        seen = set()
        layers = []
        while stack:
            expr = stack.pop()

            if expr._name in seen:
                continue
            seen.add(expr._name)

            layers.append(expr._layer())
            for operand in expr.operands:
                if isinstance(operand, Expr):
                    stack.append(operand)

        return toolz.merge(layers)

    def __dask_keys__(self):
        return [(self._name, i) for i in range(self.npartitions)]

    def __dask_postcompute__(self):
        return _concat, ()

    def __dask_postpersist__(self):
        return from_graph, (self._meta, self.divisions, self._name)


class Blockwise(Expr):
    """Super-class for block-wise operations

    This is fairly generic, and includes definitions for `_meta`, `divisions`,
    `_layer` that are often (but not always) correct.  Mostly this helps us
    avoid duplication in the future.
    """

    operation = None

    @property
    def _meta(self):
        return self.operation(
            *[arg._meta if isinstance(arg, Expr) else arg for arg in self.operands]
        )

    @property
    def _kwargs(self):
        return {}

    def _divisions(self):
        # This is an issue.  In normal Dask we re-divide everything in a step
        # which combines divisions and graph.
        # We either have to create a new Align layer (ok) or combine divisions
        # and graph into a single operation.
        first = [o for o in self.operands if isinstance(o, Expr)][0]
        assert all(
            arg.divisions == first.divisions
            for arg in self.operands
            if isinstance(arg, Expr)
        )
        return first.divisions

    @property
    def _name(self):
        return funcname(self.operation) + "-" + tokenize(*self.operands)

    def _layer(self):
        return {
            (self._name, i): (
                apply,
                self.operation,
                [
                    (operand._name, i) if isinstance(operand, Expr) else operand
                    for operand in self.operands
                ],
                self._kwargs,
            )
            for i in range(self.npartitions)
        }


class Elemwise(Blockwise):
    """
    This doesn't really do anything, but we anticipate that future
    optimizations, like `len` will care about which operations preserve length
    """

    pass


class AsType(Elemwise):
    """A good example of writing a trivial blockwise operation"""

    _parameters = ["frame", "dtypes"]
    operation = M.astype


class Apply(Elemwise):
    """A good example of writing a less-trivial blockwise operation"""

    _parameters = ["frame", "function", "args", "kwargs"]
    _defaults = {"args": (), "kwargs": {}}
    operation = M.apply

    @property
    def _meta(self):
        return self.operand("frame")._meta.apply(self.operand("function"), *self.operand("args"), **self.operand("kwargs"))

    def _layer(self):
        return {
            (self._name, i): (
                apply,
                M.apply,
                [(self.operand("frame")._name, i), self.operand("function")] + list(self.operand("args")),
                self.operand("kwargs"),
            )
            for i in range(self.npartitions)
        }


class Filter(Blockwise):
    _parameters = ["frame", "predicate"]
    operation = operator.getitem

    @classmethod
    def _replacement_rules(self):
        df = Wildcard.dot("df")
        condition = Wildcard.dot("condition")
        columns = Wildcard.dot("columns")

        # Project columns down through dataframe
        # df[df.x > 1].y -> df.y[df.x > 1]
        yield ReplacementRule(
            Pattern(Filter(df, condition)[columns]),
            lambda df, condition, columns: df[columns][condition],
        )


class Projection(Elemwise):
    """Column Selection"""

    _parameters = ["frame", "columns"]
    operation = operator.getitem

    def _divisions(self):
        return self.operand("frame").divisions

    @property
    def _meta(self):
        return self.operand("frame")._meta[self.columns]

    def _layer(self):
        return {
            (self._name, i): (operator.getitem, (self.operand("frame")._name, i), self.columns)
            for i in range(self.npartitions)
        }

    def __str__(self):
        base = str(self.operand("frame"))
        if " " in base:
            base = "(" + base + ")"
        return f"{base}[{repr(self.columns)}]"


class Index(Elemwise):
    """Column Selection"""

    _parameters = ["frame"]
    operation = getattr

    def _divisions(self):
        return self.operand("frame").divisions

    @property
    def _meta(self):
        return self.operand("frame")._meta.index

    def _layer(self):
        return {
            (self._name, i): (getattr, (self.operand("frame")._name, i), "index")
            for i in range(self.npartitions)
        }


class Binop(Elemwise):
    _parameters = ["left", "right"]
    arity = Arity.binary

    def _layer(self):
        return {
            (self._name, i): (
                self.operation,
                (self.operand("left")._name, i) if isinstance(self.operand("left"), Expr) else self.operand("left"),
                (self.operand("right")._name, i) if isinstance(self.operand("right"), Expr) else self.operand("right"),
            )
            for i in range(self.npartitions)
        }

    def __str__(self):
        return f"{self.operand('left')} {self._operator_repr} {self.operand('right')}"

    @classmethod
    def _replacement_rules(cls):
        left = Wildcard.dot("left")
        right = Wildcard.dot("right")
        columns = Wildcard.dot("columns")

        # Column Projection
        def transform(left, right, columns, cls=None):
            if isinstance(left, Expr):
                left = left[columns]  # TODO: filter just the correct columns

            if isinstance(right, Expr):
                right = right[columns]

            return cls(left, right)

        # (a + b)[c] -> a[c] + b[c]
        yield ReplacementRule(
            Pattern(cls(left, right)[columns]), functools.partial(transform, cls=cls)
        )


class Add(Binop):
    operation = operator.add
    _operator_repr = "+"

    @classmethod
    def _replacement_rules(cls):
        x = Wildcard.dot("x")
        yield ReplacementRule(
            Pattern(Add(x, x)),
            lambda x: Mul(2, x),
        )

        yield from super()._replacement_rules()


class Sub(Binop):
    operation = operator.sub
    _operator_repr = "-"


class Mul(Binop):
    operation = operator.mul
    _operator_repr = "*"

    @classmethod
    def _replacement_rules(cls):
        a, b, c = map(Wildcard.dot, "abc")
        yield ReplacementRule(
            Pattern(
                Mul(a, Mul(b, c)),
                CustomConstraint(
                    lambda a, b, c: isinstance(a, numbers.Number)
                    and isinstance(b, numbers.Number)
                ),
            ),
            lambda a, b, c: Mul(a * b, c),
        )

        yield from super()._replacement_rules()


class Div(Binop):
    operation = operator.truediv
    _operator_repr = "/"


class LT(Binop):
    operation = operator.lt
    _operator_repr = "<"


class LE(Binop):
    operation = operator.le
    _operator_repr = "<="


class GT(Binop):
    operation = operator.gt
    _operator_repr = ">"


class GE(Binop):
    operation = operator.ge
    _operator_repr = ">="


class EQ(Binop):
    operation = operator.eq
    _operator_repr = "=="


class NE(Binop):
    operation = operator.ne
    _operator_repr = "!="


class IO(Expr):
    pass


class ReadCSV(IO):
    _parameters = ["filename", "usecols", "header"]
    _defaults = {"usecols": None, "header": None}


class from_pandas(IO):
    """The only way today to get a real dataframe"""

    _parameters = ["frame", "npartitions"]
    _defaults = {"npartitions": 1}

    @property
    def _meta(self):
        return self.operand("frame").head(0)

    def _divisions(self):
        return [None] * (self.npartitions + 1)

    def _layer(self):
        chunksize = int(math.ceil(len(self.operand("frame")) / self.npartitions))
        locations = list(range(0, len(self.operand("frame")), chunksize)) + [len(self.operand("frame"))]
        return {
            (self._name, i): self.operand("frame").iloc[start:stop]
            for i, (start, stop) in enumerate(zip(locations[:-1], locations[1:]))
        }

    def __str__(self):
        return "df"

    __repr__ = __str__


class from_graph(IO):
    """A DataFrame created from an opaque Dask task graph

    This is used in persist, for example, and would also be used in any
    conversion from legacy dataframes.
    """

    _parameters = ["layer", "_meta", "divisions", "_name"]

    def _layer(self):
        return self.operand("layer")


@normalize_token.register(Expr)
def normalize_expression(expr):
    return expr._name


def optimize(expr):
    """High level query optimization

    Today we just use MatchPy's term rewriting system, leveraging the
    replacement rules found in the `replacement_rules` global list .  We continue
    rewriting until nothing changes.  The `replacement_rules` list can be added
    to by anyone, but is mostly populated by the various `_replacement_rules`
    methods on the Expr subclasses.

    Note: matchpy expects `__eq__` and `__ne__` to work in a certain way during
    matching.  This is a bit of a hack, but we turn off our implementations of
    `__eq__` and `__ne__` when this function is running using the
    `_defer_to_matchpy` global.  Please forgive us our sins, as we forgive
    those who sin against us.
    """
    last = None
    global _defer_to_matchpy

    _defer_to_matchpy = True  # take over ==/!= when optimizing
    try:
        while str(expr) != str(last):
            last = expr
            expr = replace_all(expr, replacement_rules)
    finally:
        _defer_to_matchpy = False
    return expr


from dask_match.reductions import Count, Max, Min, Mode, Size, Sum
