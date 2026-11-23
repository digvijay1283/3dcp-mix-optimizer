"""
Trains and evaluates 28-day compressive strength models on the cleaned 3DCP mix table.

Run from the project root:
    python ml/prepare_data.py                      # once, builds data_clean/strength_28d.csv
    python ml/train_strength.py                    # compare targets and models, no files written
    python ml/train_strength.py --target cast --model rf --save

Validation is repeated grouped K-fold with the source paper as the group: mixes from the same
paper share materials and test protocol, so a random split would leak and overstate accuracy.
Hyperparameters are fixed (not tuned on the validation folds).
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so the saved pipeline references engine.strength, importable from the app

from engine.strength import FEATURES, log_wb  # noqa: E402

DATA_CSV = ROOT / "data_clean" / "strength_28d.csv"
MODEL_DIR = ROOT / "model"

N_FOLDS, N_REPEATS, SEED = 5, 10, 0


def make_model(name: str):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Pipeline([("log", FunctionTransformer(log_wb)), ("scale", StandardScaler()), ("m", Ridge(alpha=1.0))])
    if name == "rf":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.7, random_state=SEED, n_jobs=-1)
    if name == "gbr":
        return GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=3, subsample=0.8,
                                         min_samples_leaf=4, random_state=SEED)
    raise ValueError(name)


MODELS = ["mean", "ridge", "rf", "gbr"]


def grouped_splits(groups: np.ndarray, rng: np.random.Generator):
    """One shuffled grouped K-fold: whole papers go to either train or validation."""
    uniq = rng.permutation(np.unique(groups))
    fold_of = {g: i % N_FOLDS for i, g in enumerate(uniq)}
    folds = np.array([fold_of[g] for g in groups])
    for k in range(N_FOLDS):
        yield np.where(folds != k)[0], np.where(folds == k)[0]


def cross_validate(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    rng = np.random.default_rng(SEED)
    r2s, maes, rmses = [], [], []
    oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))
    for _ in range(N_REPEATS):
        pred = np.zeros(len(y))
        for tr, va in grouped_splits(groups, rng):
            model = make_model(model_name).fit(X[tr], y[tr])
            pred[va] = model.predict(X[va])
        r2s.append(r2_score(y, pred))
        maes.append(mean_absolute_error(y, pred))
        rmses.append(mean_squared_error(y, pred) ** 0.5)
        oof_sum += pred
        oof_cnt += 1
    return {
        "r2": float(np.mean(r2s)), "r2_sd": float(np.std(r2s)),
        "mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
        "oof": oof_sum / oof_cnt,
    }


def load_target(target: str):
    df = pd.read_csv(DATA_CSV)
    df = df[df[f"fck_{target}"].notna()].reset_index(drop=True)
    return df, df[FEATURES].to_numpy(float), df[f"fck_{target}"].to_numpy(float), df["source"].to_numpy()


def compare(targets):
    print(f"{'target':15s}{'n':>5s}{'papers':>8s}  {'model':6s}{'R2':>8s}{'+-sd':>7s}{'MAE':>8s}{'RMSE':>8s}   (grouped {N_FOLDS}-fold x{N_REPEATS})")
    for target in targets:
        df, X, y, g = load_target(target)
        for name in MODELS:
            r = cross_validate(name, X, y, g)
            print(f"{target:15s}{len(y):5d}{len(set(g)):8d}  {name:6s}{r['r2']:8.3f}{r['r2_sd']:7.3f}{r['mae']:8.2f}{r['rmse']:8.2f}")
        print()


def sanity_checks(model, X, y):
    """Physical plausibility of the fitted model on a median mix."""
    med = np.median(X, axis=0)
    wbs = np.array([0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.45])
    grid = np.tile(med, (len(wbs), 1))
    grid[:, 0] = wbs
    print("\nSanity: predicted fck vs W/B on a median mix (should fall as W/B rises):")
    for w, p in zip(wbs, model.predict(grid)):
        print(f"  W/B {w:.2f} -> {p:5.1f} MPa")
    grid = np.tile(med, (4, 1))
    grid[:, 2] = [0.0, 0.05, 0.10, 0.15]
    print("Predicted fck vs silica fume fraction (median mix):",
          ", ".join(f"{s:.2f}->{p:.1f}" for s, p in zip(grid[:, 2], model.predict(grid))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["cast", "printed", "representative"])
    ap.add_argument("--model", choices=MODELS[1:], default="rf")
    ap.add_argument("--save", action="store_true", help="fit on all data and write model/strength_pipeline.joblib")
    args = ap.parse_args()

    if not args.target:
        compare(["cast", "printed", "representative"])
        return

    df, X, y, g = load_target(args.target)
    res = cross_validate(args.model, X, y, g)
    print(f"{args.target} / {args.model}: n={len(y)}, papers={len(set(g))}, "
          f"R2={res['r2']:.3f}+-{res['r2_sd']:.3f}, MAE={res['mae']:.2f} MPa, RMSE={res['rmse']:.2f} MPa")

    in_range = df["wb"].between(0.16, 0.26).to_numpy()
    if in_range.sum() >= 5:
        mae_in = mean_absolute_error(y[in_range], res["oof"][in_range])
        print(f"Out-of-fold MAE inside the app's W/B range 0.16-0.26: {mae_in:.2f} MPa (n={in_range.sum()})")

    model = make_model(args.model).fit(X, y)
    sanity_checks(model, X, y)

    if args.save:
        MODEL_DIR.mkdir(exist_ok=True)
        meta = {
            "target": f"28-day compressive strength, {args.target} specimens [MPa]",
            "features": FEATURES,
            "model": args.model,
            "n_train": int(len(y)),
            "n_papers": int(len(set(g))),
            "cv_r2": res["r2"], "cv_mae_mpa": res["mae"], "cv_rmse_mpa": res["rmse"],
            "train_feature_range": {f: [float(X[:, i].min()), float(X[:, i].max())] for i, f in enumerate(FEATURES)},
            "source": "Monkman, Dataset of 3D Printable Concrete Mix Designs, Zenodo 10.5281/zenodo.15333235 (CC BY 4.0)",
        }
        joblib.dump({"model": model, "meta": meta}, MODEL_DIR / "strength_pipeline.joblib")
        (MODEL_DIR / "strength_metrics.json").write_text(json.dumps(meta, indent=2))
        print(f"\nSaved {MODEL_DIR / 'strength_pipeline.joblib'}")


if __name__ == "__main__":
    main()
