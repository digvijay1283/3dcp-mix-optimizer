"""
Data-driven 28-day compressive strength estimate (M3.3), trained on published 3DCP mixes.

The model lives in model/strength_pipeline.joblib and is built by ml/train_strength.py.
It is a literature-data regression (many labs, mixed test protocols), so treat outputs as
indicative. Inputs are normalised mix descriptors the app can derive from its batch sheet.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "strength_pipeline.joblib"

# Order matters: the first column (W/B) is log-transformed inside the pipeline.
FEATURES = ["wb", "fa_frac", "sf_frac", "ab", "sp_pct", "binder_kg_m3"]


def log_wb(X):
    """Log-transform the W/B column (strength falls roughly log-linearly with W/B)."""
    X = np.array(X, dtype=float, copy=True)
    X[:, 0] = np.log(X[:, 0])
    return X


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    """Returns the {'model', 'meta'} bundle, or None if no trained model has been saved."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def features_from_mix(weights_kg: dict, binder_kg: float, wb: float, sp_pct: float) -> list:
    """Builds the model's feature vector from the app's solid-weight batch sheet (kg per m3)."""
    aggregate = weights_kg["fine_sand"] + weights_kg["medium_sand"]
    return [
        wb,
        weights_kg["fly_ash"] / binder_kg,
        weights_kg["silica_fume"] / binder_kg,
        aggregate / binder_kg,
        sp_pct,
        binder_kg,
    ]


def predict_fck_mpa(bundle: dict, features: list) -> dict:
    """Predicts fck and lists any features that fall outside the range the model was trained on."""
    pred = max(float(bundle["model"].predict([features])[0]), 0.0)
    ranges = bundle["meta"]["train_feature_range"]
    outside = [f for f, v in zip(FEATURES, features) if not ranges[f][0] <= v <= ranges[f][1]]
    return {
        "fck_mpa": pred,
        "mae_mpa": bundle["meta"]["cv_mae_mpa"],
        "outside_training_range": outside,
    }
