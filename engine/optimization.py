"""
Particle packing optimization: single-curve and multilevel MAA solvers.

Both solvers fit the same 5 material PSD curves (OPC, Fly Ash, Silica Fume,
Fine Sand, Medium Sand) against a target gradation curve, solving for the
volume fraction of each material that minimizes squared error between the
blended mix curve and the target curve.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_DATA_PATH = Path(__file__).parent / "psd_data.json"

MATERIAL_KEYS = ["opc_53", "fly_ash", "silica_fume", "fine_sand", "medium_sand"]


def load_psd_data():
    with open(_DATA_PATH, "r") as f:
        return json.load(f)


def _material_matrix(psd_data):
    """Returns (sieve_sizes, matrix) where matrix is shape (n_sieves, n_materials),
    each column a material's cumulative % passing, as a fraction (0-1)."""
    sieves = np.array(psd_data["sieve_diameters_um"], dtype=float)
    matrix = np.column_stack(
        [np.array(psd_data["materials"][k]["psd_passing"], dtype=float) / 100.0 for k in MATERIAL_KEYS]
    )
    return sieves, matrix


def _fuller_target(sieves, d_max, q):
    """Single-curve Fuller/Andreasen-Andersen target: P(D) = (D/Dmax)^q."""
    d_max_arr = np.full_like(sieves, d_max)
    ratio = np.clip(sieves / d_max_arr, 0, 1)
    return np.power(ratio, q)


def _multilevel_target(sieves, d_max, p, q1, q2, d_gap_lo=90.0, d_gap_hi=150.0):
    """
    Piecewise multilevel MAA target curve with a flat gap step at breakpoint p. Each branch
    uses the Modified A&A form P(D) = (D^q - Dmin^q) / (Dmax^q - Dmin^q) with its own Dmin/Dmax:
      - below the gap (D <= d_gap_lo): powder sub-curve (modulus q1) from d_min up to p at d_gap_lo
      - in the gap (d_gap_lo < D < d_gap_hi): flat plateau at value p (the gap-gradation step)
      - above the gap (D >= d_gap_hi): aggregate sub-curve (modulus q2) from p at d_gap_hi to 1.0 at d_max

    d_min is the smallest sieve. Omitting it (a plain (D/Dmax)^q) leaves the target at ~0.25 for
    sub-micron sizes, which no real powder can deliver and drags R2 down to ~0.7.
    """
    target = np.zeros_like(sieves)
    d_min = float(np.min(sieves))

    def maa(d, lo, hi, q):
        return np.clip((d ** q - lo ** q) / (hi ** q - lo ** q), 0.0, 1.0)

    powder_mask = sieves <= d_gap_lo
    gap_mask = (sieves > d_gap_lo) & (sieves < d_gap_hi)
    aggregate_mask = sieves >= d_gap_hi

    target[powder_mask] = p * maa(sieves[powder_mask], d_min, d_gap_lo, q1)
    target[gap_mask] = p
    target[aggregate_mask] = p + (1.0 - p) * maa(sieves[aggregate_mask], d_gap_hi, d_max, q2)

    return np.clip(target, 0, 1)


#   Mix-design-realistic bounds on each material's volume fraction of total
#   solids (not just [0,1]). Without these, the unconstrained least-squares
#   fit happily over-uses silica fume (its ultra-fine PSD fits gap-gradation
#   targets "too well" mathematically) at proportions no real 3DCP mix uses.
#   Order matches MATERIAL_KEYS: opc_53, fly_ash, silica_fume, fine_sand, medium_sand
FRACTION_BOUNDS_LO = np.array([0.15, 0.00, 0.00, 0.15, 0.05])
FRACTION_BOUNDS_HI = np.array([0.45, 0.25, 0.12, 0.55, 0.40])


def _solve_fractions(sieves, matrix, target):
    """
    Least-squares solve for material volume fractions v (sum(v) = 1, v within
    realistic per-material bounds) minimizing || matrix @ v - target ||^2.
    """
    n_materials = matrix.shape[1]

    def residuals(v):
        blended = matrix @ v
        curve_resid = blended - target
        # soft constraint: fractions should sum to 1
        sum_resid = [10.0 * (np.sum(v) - 1.0)]
        return np.concatenate([curve_resid, sum_resid])

    v0 = (FRACTION_BOUNDS_LO + FRACTION_BOUNDS_HI) / 2.0
    v0 = v0 / np.sum(v0)
    bounds = (FRACTION_BOUNDS_LO, FRACTION_BOUNDS_HI)

    result = least_squares(residuals, v0, bounds=bounds, method="trf")

    v = result.x
    v = np.clip(v, 0, None)
    v = v / np.sum(v)  # renormalize to exactly sum to 1

    blended = matrix @ v
    rss = float(np.sum((blended - target) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 1e-12 else 1.0

    return v, rss, r2, blended


def optimize_single_curve(psd_data, q_single, d_max=2360.0):
    """Conventional single-curve MAA optimization."""
    sieves, matrix = _material_matrix(psd_data)
    target = _fuller_target(sieves, d_max, q_single)
    v, rss, r2, blended = _solve_fractions(sieves, matrix, target)
    return {
        "fractions": dict(zip(MATERIAL_KEYS, v.tolist())),
        "rss": rss,
        "r2": r2,
        "target_curve": target.tolist(),
        "blended_curve": blended.tolist(),
    }


def optimize_multilevel(psd_data, p, q1, q2, d_max=2360.0, d_gap_lo=90.0, d_gap_hi=150.0):
    """Piecewise multilevel MAA optimization with the flat gap step at p."""
    sieves, matrix = _material_matrix(psd_data)
    target = _multilevel_target(sieves, d_max, p, q1, q2, d_gap_lo, d_gap_hi)
    v, rss, r2, blended = _solve_fractions(sieves, matrix, target)
    return {
        "fractions": dict(zip(MATERIAL_KEYS, v.tolist())),
        "rss": rss,
        "r2": r2,
        "target_curve": target.tolist(),
        "blended_curve": blended.tolist(),
    }


def run_dual_optimization(psd_data, p, q1, q2, q_single, d_max=2360.0):
    """Runs both solvers and returns both result sets plus the shared sieve axis."""
    sieves, _ = _material_matrix(psd_data)
    single = optimize_single_curve(psd_data, q_single, d_max)
    multilevel = optimize_multilevel(psd_data, p, q1, q2, d_max)
    return {
        "sieves_um": sieves.tolist(),
        "single": single,
        "multilevel": multilevel,
    }
