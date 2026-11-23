"""
Cleans the Monkman 3D-printable-concrete mix dataset (Zenodo 10.5281/zenodo.15333235)
into a small, in-domain table for strength modelling.

Run from the project root:  python ml/prepare_data.py

Domain rules (so the model only sees mixes the app can actually describe):
  - 28-day strengths only
  - binder is cement, optionally with fly ash and/or silica fume (no slag, limestone, CSA, ...)
  - no fibers, no special concrete types (lightweight, ECC, UHPFRC, ...)
  - physically sane W/B and aggregate/binder ratio

Features are ones the app can compute from its own batch sheet. Missing material
quantities are read as "not used" (0 kg/m3), per the dataset's entered-quantity convention.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data_raw" / "zenodo_mixes_1149" / "S4_3DPC_mix_details_1149.csv"
OUT_CSV = ROOT / "data_clean" / "strength_28d.csv"

ALLOWED_BINDERS = {"CEM", "CEM-FA", "CEM-SFM", "CEM-FA-SFM"}
MAX_WB = 0.60  # above this the mix is a foamed/lightweight outlier, far from any printable mortar
MAX_AGG_BINDER = 5.0

FEATURES = ["wb", "fa_frac", "sf_frac", "ab", "sp_pct", "binder_kg_m3"]
TARGETS = {
    "cast": "Strength - cast [MPa]",
    "printed": "Strength - printed [MPa]",
    "representative": "Representative strength [MPa]",
}


def load_raw() -> pd.DataFrame:
    # "-" and "NR" (not reported) mark missing values in numeric columns
    return pd.read_csv(RAW_CSV, encoding="utf-8-sig", na_values=["-", "NR", ""])


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def clean(df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    steps = [("raw rows", len(df))]

    def keep(mask, label):
        nonlocal df
        df = df[mask(df)].copy()
        steps.append((label, len(df)))

    keep(lambda d: num(d, "Strength age [days]") == 28, "28-day strength")
    keep(lambda d: d["Binder"].isin(ALLOWED_BINDERS), "binder = cement (+ fly ash / silica fume)")
    keep(lambda d: num(d, "Fibers [kg/m3]").fillna(0) == 0, "no fibers")
    keep(lambda d: d["Special Type"].isna(), "no special concrete type")

    binder = num(df, "Total Binder [kg/m3]")
    water = num(df, "Water [kg/m3]")
    agg = num(df, "Total Aggregate [kg/m3]")
    out = pd.DataFrame({
        "source": df["Source"].values,
        "mix_id": df["ID"].values,
        "binder_kg_m3": binder.values,
        "wb": (water / binder).values,
        "fa_frac": (num(df, "Fly Ash [kg/m3]").fillna(0) / binder).values,
        "sf_frac": (num(df, "Silica Fume [kg/m3]").fillna(0) / binder).values,
        "ab": (agg / binder).values,
        "sp_pct": (num(df, "Plasticizer [kg/m3]").fillna(0) / binder * 100).values,
    })
    for name, col in TARGETS.items():
        out[f"fck_{name}"] = num(df, col).values

    def keep_out(mask, label):
        nonlocal out
        out = out[mask(out)].copy()
        steps.append((label, len(out)))

    keep_out(lambda d: d[FEATURES].notna().all(axis=1), "all features present")
    keep_out(lambda d: d["wb"].between(0.10, MAX_WB), f"W/B within 0.10-{MAX_WB}")
    keep_out(lambda d: d["ab"].between(0.2, MAX_AGG_BINDER), f"aggregate/binder within 0.2-{MAX_AGG_BINDER}")
    keep_out(lambda d: d["sf_frac"].between(0, 0.4) & d["fa_frac"].between(0, 0.8), "silica fume <= 40%, fly ash <= 80% of binder")
    keep_out(lambda d: d["sp_pct"].between(0, 8), "plasticizer <= 8% of binder")
    keep_out(lambda d: d[[f"fck_{t}" for t in TARGETS]].notna().any(axis=1), "at least one strength value")
    before = len(out)
    out = out.drop_duplicates(subset=FEATURES + [f"fck_{t}" for t in TARGETS]).reset_index(drop=True)
    steps.append((f"duplicates removed ({before - len(out)})", len(out)))

    if log:
        for label, n in steps:
            print(f"{n:6d}  {label}")
    return out


if __name__ == "__main__":
    cleaned = clean(load_raw())
    OUT_CSV.parent.mkdir(exist_ok=True)
    cleaned.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(cleaned)} rows, {cleaned['source'].nunique()} source papers)")
    for name in TARGETS:
        col = f"fck_{name}"
        print(f"  {name:15s} n={cleaned[col].notna().sum():4d}  range {cleaned[col].min():.0f}-{cleaned[col].max():.0f} MPa")
    print("\nFeature summary:")
    print(cleaned[FEATURES].describe().T[["count", "min", "50%", "max"]].round(3).to_string())
    low_wb = cleaned["wb"].between(0.16, 0.26).sum()
    print(f"\nMixes inside the app's W/B slider range (0.16-0.26): {low_wb} of {len(cleaned)}")
