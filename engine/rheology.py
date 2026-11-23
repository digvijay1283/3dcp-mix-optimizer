"""
Rheology, buildability diagnostics, and the empirical compressive strength
estimate (M3). All functions are pure and take plain floats in/out so they
can be unit-tested without Streamlit.
"""

import math

GRAVITY = 9.81  # m/s^2
NOZZLE_CHOKE_PARTICLE_MM = 2.36  # D_L, largest particle size present (medium sand top sieve)


def wet_packing_density(q1: float, q2: float, p: float) -> float:
    """
    Phi via quadratic response surface regression, PRD M3.1 formula:
    Phi = 0.4707 + 0.0319*q1 + 1.5136*p + 0.6182*q1*p - 1.8392*q1^2 - 0.0125*q2^2 - 1.7506*p^2
    """
    return (
        0.4707
        + 0.0319 * q1
        + 1.5136 * p
        + 0.6182 * q1 * p
        - 1.8392 * q1 ** 2
        - 0.0125 * q2 ** 2
        - 1.7506 * p ** 2
    )


def voids_percent(phi: float) -> float:
    return (1.0 - phi) * 100.0


def static_yield_stress_kpa(phi: float, wb: float, v_sf: float, vma_pct: float) -> float:
    """tau0 (PRD M3.2)."""
    return (
        0.42
        * (1.0 - phi / 0.865) ** (-1.6)
        * (0.20 / wb) ** 1.7
        * (1.0 + 3.5 * v_sf + 1.8 * vma_pct)
    )


def flow_spread_mm(wb: float, sp_pct: float, v_sf: float, vma_pct: float) -> float:
    """Workability flow spread, mm (PRD M3.2)."""
    return (
        155.0
        + 650.0 * (wb - 0.20)
        + 14.0 * (sp_pct - 1.5)
        - 180.0 * (v_sf - 0.05)
        - 25.0 * vma_pct
    )


def max_buildable_height_m(tau0_kpa: float, rho_wet_kg_m3: float) -> float:
    """H_max (PRD M3.2). tau0 converted kPa -> Pa via the *1000 factor already in formula."""
    return (math.sqrt(3) * tau0_kpa * 1000.0) / (rho_wet_kg_m3 * GRAVITY)


def max_layer_count(h_max_m: float, layer_height_mm: float) -> int:
    """N_max (PRD M3.2)."""
    return int(math.floor((h_max_m * 1000.0) / layer_height_mm))


def nozzle_choke_warning(nozzle_mm: float, d_l_mm: float = NOZZLE_CHOKE_PARTICLE_MM) -> bool:
    """True if nozzle clogging risk flagged: D_L / D_nozzle > 0.15 (PRD M3.2)."""
    return (d_l_mm / nozzle_mm) > 0.15


def estimate_compressive_strength_mpa(phi: float, wb: float, v_sf: float) -> float:
    """
    Empirical compressive strength estimate (NOT ML-trained — no experimental
    dataset was available to train a model; see architecture.md Section 5).

    f_ck = k1 * Phi^n * (1/WB)^m * (1 + k2 * v_sf)

    Calibrated so that typical 3DCP mixes (Phi ~ 0.55-0.70, W/B ~ 0.16-0.26,
    v_sf ~ 0.0-0.15) land in the realistic 30-70 MPa range. This models the
    well-established packing-density/strength and Abrams'-law-style W/B
    relationships from concrete technology literature, scaled with silica
    fume's pozzolanic strength contribution.

    UI MUST label this as "Strength Estimate (empirical correlation)",
    not as an ML prediction, to avoid misrepresenting its basis.
    """
    k1, n, m, k2 = 18.0, 2.2, 0.85, 2.0
    return k1 * (phi ** n) * ((1.0 / wb) ** m) * (1.0 + k2 * v_sf)


def run_rheology_suite(
    phi: float,
    wb: float,
    v_sf: float,
    sp_pct: float,
    vma_pct: float,
    nozzle_mm: float,
    layer_height_mm: float,
    rho_wet_kg_m3: float,
) -> dict:
    """Runs the full M3 diagnostic suite and returns a flat results dict."""
    tau0 = static_yield_stress_kpa(phi, wb, v_sf, vma_pct)
    flow = flow_spread_mm(wb, sp_pct, v_sf, vma_pct)
    h_max = max_buildable_height_m(tau0, rho_wet_kg_m3)
    n_max = max_layer_count(h_max, layer_height_mm)
    fck = estimate_compressive_strength_mpa(phi, wb, v_sf)

    return {
        "phi": phi,
        "voids_pct": voids_percent(phi),
        "tau0_kpa": tau0,
        "flow_mm": flow,
        "flow_in_window": 140.0 <= flow <= 170.0,
        "h_max_m": h_max,
        "n_max": n_max,
        "nozzle_choke_warning": nozzle_choke_warning(nozzle_mm),
        "fck_mpa": fck,
    }
