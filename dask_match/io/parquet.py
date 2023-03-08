from __future__ import annotations

from functools import cached_property, partial

import dask
from dask.dataframe.io.parquet.core import (
    ParquetFunctionWrapper,
    get_engine,
    process_statistics,
    set_index_columns,
)
from dask.dataframe.io.parquet.utils import _split_user_options
from dask.utils import natural_sort_key
from fsspec.utils import stringify_path
from matchpy import CustomConstraint, Pattern, ReplacementRule, Wildcard

from dask_match.core import EQ, GE, GT, IO, LE, LT, NE, Filter

NONE_LABEL = "__null_dask_index__"


def _list_columns(columns):
    # Simple utility to convert columns to list
    if isinstance(columns, (str, int)):
        columns = [columns]
    elif isinstance(columns, tuple):
        columns = list(columns)
    return columns


class ReadParquet(IO):
    """Read a parquet dataset"""

    _parameters = [
        "path",
        "columns",
        "filters",
        "categories",
        "index",
        "storage_options",
        "use_nullable_dtypes",
        "calculate_divisions",
        "ignore_metadata_file",
        "metadata_task_size",
        "split_row_groups",
        "blocksize",
        "aggregate_files",
        "parquet_file_extension",
        "filesystem",
        "kwargs",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "categories": None,
        "index": None,
        "storage_options": None,
        "use_nullable_dtypes": False,
        "calculate_divisions": False,
        "ignore_metadata_file": False,
        "metadata_task_size": None,
        "split_row_groups": "infer",
        "blocksize": "default",
        "aggregate_files": None,
        "parquet_file_extension": (".parq", ".parquet", ".pq"),
        "filesystem": "fsspec",
        "kwargs": {},
    }

    @property
    def engine(self):
        return get_engine("pyarrow")

    @property
    def columns(self):
        if self.operand("columns") is None:
            return self._meta.columns
        else:
            import pandas as pd

            return pd.Index(_list_columns(self.operand("columns")))

    @classmethod
    def _replacement_rules(cls):
        _ = Wildcard.dot()
        a, b, c, d, e, f = map(Wildcard.dot, "abcdef")

        # Column projection
        yield ReplacementRule(
            Pattern(ReadParquet(a, columns=b, filters=c)[d]),
            lambda a, b, c, d: ReadParquet(a, columns=_list_columns(d), filters=c),
        )

        # Simple dict to make sure field comes first in filter
        flip_op = {LE: GE, LT: GT, GE: LE, GT: LT}

        # Predicate pushdown to parquet
        for op in [LE, LT, GE, GT, EQ, NE]:

            def predicate_pushdown(a, b, c, d, e, op=None):
                return ReadParquet(
                    a, columns=_list_columns(b), filters=(c or []) + [(d, op._operator_repr, e)]
                )

            yield ReplacementRule(
                Pattern(
                    Filter(
                        ReadParquet(a, columns=b, filters=c),
                        op(ReadParquet(a, columns=_, filters=_)[d], e),
                    )
                ),
                partial(predicate_pushdown, op=op),
            )

            def predicate_pushdown(a, b, c, d, e, op=None):
                return ReadParquet(
                    a, columns=_list_columns(b), filters=(c or []) + [(d, op._operator_repr, e)]
                )

            yield ReplacementRule(
                Pattern(
                    Filter(
                        ReadParquet(a, columns=b, filters=c),
                        op(e, ReadParquet(a, columns=_, filters=_)[d]),
                    )
                ),
                partial(predicate_pushdown, op=flip_op.get(op, op)),
            )

            def predicate_pushdown(a, b, c, d, e, op=None):
                return ReadParquet(
                    a, columns=_list_columns(b), filters=(c or []) + [(d, op._operator_repr, e)]
                )

            yield ReplacementRule(
                Pattern(
                    Filter(
                        ReadParquet(a, columns=b, filters=c),
                        op(ReadParquet(a, columns=d, filters=_), e),
                    ),
                    CustomConstraint(lambda d: isinstance(d, str)),
                ),
                partial(predicate_pushdown, op=op),
            )

            def predicate_pushdown(a, b, c, d, e, op=None):
                return ReadParquet(
                    a, columns=_list_columns(b), filters=(c or []) + [(d, op._operator_repr, e)]
                )

            yield ReplacementRule(
                Pattern(
                    Filter(
                        ReadParquet(a, columns=b, filters=c),
                        op(e, ReadParquet(a, columns=d, filters=_)),
                    ),
                    CustomConstraint(lambda d: isinstance(d, str)),
                ),
                partial(predicate_pushdown, op=flip_op.get(op, op)),
            )

    @cached_property
    def _dataset_info(self):

        # Process and split user options
        (
            dataset_options,
            read_options,
            open_file_options,
            other_options,
        ) = _split_user_options(**self.operand("kwargs"))

        # Extract global filesystem and paths
        fs, paths, dataset_options, open_file_options = self.engine.extract_filesystem(
            self.operand("path"),
            self.operand("filesystem"),
            dataset_options,
            open_file_options,
            self.operand("storage_options"),
        )
        read_options["open_file_options"] = open_file_options
        paths = sorted(paths, key=natural_sort_key)  # numeric rather than glob ordering

        auto_index_allowed = False
        if self.operand("index") is None:
            # User is allowing auto-detected index
            auto_index_allowed = True
        if self.operand("index") and isinstance(self.operand("index"), str):
            index = [self.operand("index")]
        else:
            index = self.operand("index")

        blocksize = self.operand("blocksize")
        if self.operand("split_row_groups") in ("infer", "adaptive"):
            # Using blocksize to plan partitioning
            if self.operand("blocksize") == "default":
                if hasattr(self.engine, "default_blocksize"):
                    blocksize = self.engine.default_blocksize()
                else:
                    blocksize = "128MiB"
        else:
            # Not using blocksize - Set to `None`
            blocksize = None

        dataset_info = self.engine._collect_dataset_info(
            paths,
            fs,
            self.operand("categories"),
            index,
            self.operand("calculate_divisions"),
            self.operand("filters"),
            self.operand("split_row_groups"),
            blocksize,
            self.operand("aggregate_files"),
            self.operand("ignore_metadata_file"),
            self.operand("metadata_task_size"),
            self.operand("parquet_file_extension"),
            {
                "read": read_options,
                "dataset": dataset_options,
                **other_options,
            },
        )

        # Infer meta, accounting for index and columns arguments.
        meta = self.engine._create_dd_meta(dataset_info, self.operand("use_nullable_dtypes"))
        index = [index] if isinstance(index, str) else index
        meta, index, columns = set_index_columns(
            meta, index, self.operand("columns"), auto_index_allowed
        )
        if meta.index.name == NONE_LABEL:
            meta.index.name = None
        dataset_info["meta"] = meta
        dataset_info["index"] = index
        dataset_info["columns"] = columns

        return dataset_info

    @property
    def _meta(self):
        return self._dataset_info["meta"]

    @cached_property
    def _plan(self):
        dataset_info = self._dataset_info
        parts, stats, common_kwargs = self.engine._construct_collection_plan(
            dataset_info
        )

        # Parse dataset statistics from metadata (if available)
        parts, divisions, _ = process_statistics(
            parts,
            stats,
            dataset_info["filters"],
            dataset_info["index"],
            (
                dataset_info["blocksize"]
                if dataset_info["split_row_groups"] is True
                else None
            ),
            dataset_info["split_row_groups"],
            dataset_info["fs"],
            dataset_info["aggregation_depth"],
        )

        meta = dataset_info["meta"]
        if len(divisions) < 2:
            # empty dataframe - just use meta
            divisions = (None, None)
            io_func = lambda x: x
            parts = [meta]
        else:
            # Use IO function wrapper
            io_func = ParquetFunctionWrapper(
                self.engine,
                dataset_info["fs"],
                meta,
                dataset_info["columns"],
                dataset_info["index"],
                self.operand("use_nullable_dtypes"),
                {},  # All kwargs should now be in `common_kwargs`
                common_kwargs,
            )

        return {
            "func": io_func,
            "parts": parts,
            "divisions": divisions,
        }

    def _divisions(self):
        return self._plan["divisions"]

    def _layer(self):
        io_func = self._plan["func"]
        parts = self._plan["parts"]
        return {(self._name, i): (io_func, part) for i, part in enumerate(parts)}


def read_parquet(
    path=None,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    use_nullable_dtypes=False,
    calculate_divisions=False,
    ignore_metadata_file=False,
    metadata_task_size=None,
    split_row_groups="infer",
    blocksize="default",
    aggregate_files=None,
    parquet_file_extension=(".parq", ".parquet", ".pq"),
    filesystem="fsspec",
    **kwargs,
):
    if use_nullable_dtypes:
        use_nullable_dtypes = dask.config.get("dataframe.dtype_backend")

    if hasattr(path, "name"):
        path = stringify_path(path)

    return ReadParquet(
        path,
        columns=_list_columns(columns),
        filters=filters,
        categories=categories,
        index=index,
        storage_options=storage_options,
        use_nullable_dtypes=use_nullable_dtypes,
        calculate_divisions=calculate_divisions,
        ignore_metadata_file=ignore_metadata_file,
        metadata_task_size=metadata_task_size,
        split_row_groups=split_row_groups,
        blocksize=blocksize,
        aggregate_files=aggregate_files,
        parquet_file_extension=parquet_file_extension,
        filesystem=filesystem,
        **kwargs,
    )
