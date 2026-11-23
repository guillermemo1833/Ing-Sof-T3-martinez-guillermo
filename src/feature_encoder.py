"""Feature encoding utilities for tabular data.

This module defines the :class:`FeatureEncoder` class, which groups small
``pandas.DataFrame`` transformations that are typically applied before
training a model: one-hot encoding, label encoding and binning of numeric
columns. None of the methods modify the input DataFrame in place.
"""

from typing import Optional, Sequence
import pandas as pd
from pandas.api import types as pdt


class FeatureEncoder:
    """Utility class for common feature-encoding operations."""

    @staticmethod
    def _require_column(df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    def one_hot(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Replace ``col`` by one indicator column per distinct category.

        The new columns are named ``f"{col}_{category}"`` and contain 0/1
        integers. They are appended at the end of the DataFrame, in sorted
        category order, and the original column is dropped.

        Raises
        ------
        KeyError
            If ``col`` is not present in ``df``.

        Notes
        -----
        The original DataFrame is not modified; a copy is returned.
        """
        self._require_column(df, col)
        categories = sorted(df[col].dropna().unique(), key=str)
        result = df.drop(columns=[col]).copy()
        for cat in categories:
            result[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        return result

    def label_encode(
        self,
        df: pd.DataFrame,
        col: str,
        mapping: Optional[dict] = None,
    ) -> pd.Series:
        """Return an integer code for each value of ``col``.

        If ``mapping`` is None, codes are assigned in sorted order of the
        distinct values (0, 1, 2, ...). If ``mapping`` is given, it must
        cover every value present in the column.

        Returns
        -------
        codes:
            ``pd.Series`` of dtype ``int64`` with the same index as ``df``
            and name ``f"{col}_code"``.

        Raises
        ------
        KeyError
            If ``col`` is not present in ``df``.
        ValueError
            If ``mapping`` is given and the column contains a value that
            is not a key of ``mapping``.
        """
        self._require_column(df, col)
        values = df[col]
        if mapping is None:
            mapping = {v: i for i, v in enumerate(sorted(values.unique(), key=str))}
        unseen = sorted({v for v in values.unique() if v not in mapping}, key=str)
        if unseen:
            raise ValueError(f"values without a code in mapping: {unseen}")
        return values.map(mapping).astype("int64").rename(f"{col}_code")

    def bin_numeric(
        self,
        df: pd.DataFrame,
        col: str,
        edges: Sequence[float],
        labels: Sequence[str],
    ) -> pd.DataFrame:
        """Add a column ``f"{col}_bin"`` with the interval label of each value.

        ``edges`` are the bin boundaries (strictly increasing) and ``labels``
        the names of the ``len(edges) - 1`` intervals. Intervals are closed
        on the right, e.g. edges ``[0, 18, 65, 120]`` and labels
        ``["young", "adult", "senior"]`` map 18 to ``"young"`` and 19 to
        ``"adult"``. Values outside the outer edges become ``NaN``.

        Raises
        ------
        KeyError
            If ``col`` is not present in ``df``.
        TypeError
            If ``col`` is not numeric.
        ValueError
            If ``edges`` is not strictly increasing or if
            ``len(labels) != len(edges) - 1``.

        Notes
        -----
        The original DataFrame is not modified; a copy is returned.
        """
        self._require_column(df, col)
        if not pdt.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric to be binned")
        edges = list(edges)
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("edges must be strictly increasing")
        if len(labels) != len(edges) - 1:
            raise ValueError("labels must have exactly len(edges) - 1 elements")

        result = df.copy()
        result[f"{col}_bin"] = pd.cut(
            result[col], bins=edges, labels=list(labels), include_lowest=True
        ).astype(object)
        return result
