import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


IPQ_REGEX = re.compile(r"\b(\d+)\s*(?:pcs|pieces|tabs|tablets|caps|capsules|ml|l|gm|g|kg|kgm|pack|packs|count)\b", re.IGNORECASE)
NUM_IN_TEXT_REGEX = re.compile(r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.lower()
    s = re.sub(r"[^a-z0-9% ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_ipq(s: str) -> float:
    if not isinstance(s, str):
        return 1.0
    # Look for explicit pack counts, otherwise fall back to max number in text
    m = IPQ_REGEX.search(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    nums = [float(x) for x in NUM_IN_TEXT_REGEX.findall(s)]
    if nums:
        return float(max(nums))
    return 1.0


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key: str):
        self.key = key

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        col = X[self.key].astype(str).fillna("")
        return col


class TextNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]):
        return [normalize_text(x) for x in X]


class NumericFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str):
        self.column = column
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, X: pd.DataFrame, y=None):
        values = self._compute(X)
        self.scaler.fit(values)
        return self

    def transform(self, X: pd.DataFrame):
        values = self._compute(X)
        return self.scaler.transform(values)

    def _compute(self, X: pd.DataFrame):
        col = X[self.column].astype(str).fillna("")
        ipq = col.map(extract_ipq).astype(float).values.reshape(-1, 1)
        has_percent = col.str.contains("%", regex=False).astype(int).values.reshape(-1, 1)
        return sparse.csr_matrix(np.hstack([ipq, has_percent]))
