"""MUTANTE M1 de metrics_utils.py: contiene exactamente un bug plantado.

No edite este archivo. Su suite de pruebas debe detectarlo (ver Tarea 3, A3).

The :class:`MetricsUtils` class provides pure NumPy helpers to evaluate
regression and classification predictions. Every method validates its
inputs and raises clear exceptions, so that tests can verify both the
normal behaviour and the error handling.
"""

from typing import Sequence
import numpy as np


class MetricsUtils:
    """Collection of basic evaluation metrics."""

    @staticmethod
    def _as_pair(y_true: Sequence, y_pred: Sequence) -> tuple[np.ndarray, np.ndarray]:
        """Convert both sequences to 1D arrays and validate their shapes.

        Raises
        ------
        ValueError
            If the sequences are empty or have different lengths.
        """
        a = np.asarray(y_true)
        b = np.asarray(y_pred)
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("inputs must be one-dimensional sequences")
        if a.size == 0:
            raise ValueError("inputs must not be empty")
        if a.shape != b.shape:
            raise ValueError(
                f"y_true and y_pred must have the same length ({a.size} != {b.size})"
            )
        return a, b

    def rmse(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """Root mean squared error: ``sqrt(mean((y_true - y_pred) ** 2))``.

        Raises
        ------
        ValueError
            If the sequences are empty or have different lengths.
        """
        a, b = self._as_pair(y_true, y_pred)
        a = a.astype(float)
        b = b.astype(float)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """Mean absolute error: ``mean(|y_true - y_pred|)``.

        Raises
        ------
        ValueError
            If the sequences are empty or have different lengths.
        """
        a, b = self._as_pair(y_true, y_pred)
        a = a.astype(float)
        b = b.astype(float)
        return float(np.mean(a - b))

    def accuracy(self, y_true: Sequence, y_pred: Sequence) -> float:
        """Fraction of positions where ``y_true`` and ``y_pred`` agree.

        Returns a value in ``[0, 1]``.

        Raises
        ------
        ValueError
            If the sequences are empty or have different lengths.
        """
        a, b = self._as_pair(y_true, y_pred)
        return float(np.mean(a == b))

    def confusion_matrix(
        self,
        y_true: Sequence,
        y_pred: Sequence,
        labels: Sequence,
    ) -> np.ndarray:
        """Confusion matrix with rows = true label and columns = predicted label.

        ``labels`` fixes the order of rows and columns. Entry ``[i, j]`` is
        the number of samples whose true label is ``labels[i]`` and whose
        predicted label is ``labels[j]``.

        Raises
        ------
        ValueError
            If the sequences are empty, have different lengths, if
            ``labels`` is empty or contains duplicates, or if any value in
            ``y_true``/``y_pred`` is not present in ``labels``.
        """
        a, b = self._as_pair(y_true, y_pred)
        labels = list(labels)
        if not labels:
            raise ValueError("labels must not be empty")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must not contain duplicates")

        index = {lab: i for i, lab in enumerate(labels)}
        unknown = sorted({v for v in list(a) + list(b) if v not in index}, key=str)
        if unknown:
            raise ValueError(f"values not present in labels: {unknown}")

        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(a, b):
            cm[index[t], index[p]] += 1
        return cm
