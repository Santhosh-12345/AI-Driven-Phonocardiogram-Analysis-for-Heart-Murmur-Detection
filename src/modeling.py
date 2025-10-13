from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

from .features import ColumnSelector, NumericFeaturizer, TextNormalizer


# SMAPE metric as a scorer

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def make_pipeline(max_features: int = 150000, alpha: float = 1.0) -> Pipeline:
    text_features = Pipeline(
        steps=[
            ("select", ColumnSelector("catalog_content")),
            ("normalize", TextNormalizer()),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    strip_accents="unicode",
                ),
            ),
        ]
    )

    numeric_features = Pipeline(
        steps=[
            ("num", NumericFeaturizer("catalog_content")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("txt", text_features, ["catalog_content"]),
            ("num", numeric_features, ["catalog_content"]),
        ],
        sparse_threshold=0.3,
    )

    model = Ridge(alpha=alpha, random_state=42)

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    return pipe


@dataclass
class TrainConfig:
    max_features: int = 200000
    alpha: float = 2.0


def train_model(df_train: pd.DataFrame, cfg: Optional[TrainConfig] = None) -> Pipeline:
    cfg = cfg or TrainConfig()
    pipeline = make_pipeline(max_features=cfg.max_features, alpha=cfg.alpha)
    y = df_train["price"].astype(float).values
    pipeline.fit(df_train, y)
    return pipeline


def predict_prices(pipe: Pipeline, df_test: pd.DataFrame) -> np.ndarray:
    preds = pipe.predict(df_test)
    preds = np.maximum(preds, 0.01)  # enforce positive prices
    return preds
