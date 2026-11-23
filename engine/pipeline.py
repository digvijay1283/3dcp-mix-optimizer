"""Runs the full M2 -> M3 -> M4 chain for one set of slider inputs."""

import numpy as np

from engine import batching, optimization, rheology, strength


def run_all(psd_data: dict, params: dict) -> dict:
    opt = optimization.run_dual_optimization(
        psd_data, params["p"], params["q1"], params["q2"], params["q_single"]
    )
    fractions = opt["multilevel"]["fractions"]

    batch = batching.run_batching_suite(
        psd_data, fractions, params["wb"], params["sp_pct"], params["vma_pct"], params["batch_l"]
    )
    rho_wet = batch["batch_1m3"]["wet_density_kg_m3"]

    phi = rheology.wet_packing_density(params["q1"], params["q2"], params["p"])
    rheo = rheology.run_rheology_suite(
        phi=phi,
        wb=params["wb"],
        v_sf=fractions["silica_fume"],
        sp_pct=params["sp_pct"],
        vma_pct=params["vma_pct"],
        nozzle_mm=params["nozzle_mm"],
        layer_height_mm=params["layer_height_mm"],
        rho_wet_kg_m3=rho_wet,
    )

    # Prefer the trained strength model; keep the empirical formula if no model file exists.
    bundle = strength.load_model()
    if bundle:
        b1 = batch["batch_1m3"]
        feats = strength.features_from_mix(b1["weights_kg"], b1["binder_mass_kg"], params["wb"], params["sp_pct"])
        est = strength.predict_fck_mpa(bundle, feats)
        rheo.update(
            fck_mpa=est["fck_mpa"], fck_source="model",
            fck_mae_mpa=est["mae_mpa"], fck_outside_range=est["outside_training_range"],
        )
    else:
        rheo.update(fck_source="formula", fck_mae_mpa=None, fck_outside_range=[])

    return {"opt": opt, "batch": batch, "rheo": rheo, "params": dict(params)}


def phi_surface(q2: float, p_range=(0.40, 0.60), q1_range=(0.01, 0.20), n=41):
    """Grid of Phi over (p, q1) at fixed q2 for the 3D surface plot."""
    p_axis = np.linspace(*p_range, n)
    q1_axis = np.linspace(*q1_range, n)
    z = np.array(
        [[rheology.wet_packing_density(q1, q2, p) for p in p_axis] for q1 in q1_axis]
    )
    return p_axis, q1_axis, z


def summary_row(result: dict) -> dict:
    """Flat dict of the active mix for CSV logging and Co-Pilot context."""
    p = result["params"]
    r = result["rheo"]
    econ = result["batch"]["economics"]
    fr = result["opt"]["multilevel"]["fractions"]
    w = result["batch"]["batch_1m3"]["weights_kg"]
    liq = result["batch"]["batch_1m3"]["liquids_l"]
    return {
        **{k: p[k] for k in ("p", "q1", "q2", "q_single", "wb", "sp_pct", "vma_pct", "nozzle_mm", "layer_height_mm", "batch_l")},
        "r2_multilevel": result["opt"]["multilevel"]["r2"],
        "r2_single": result["opt"]["single"]["r2"],
        "rss_multilevel": result["opt"]["multilevel"]["rss"],
        **{f"v_{k}": v for k, v in fr.items()},
        "phi": r["phi"],
        "voids_pct": r["voids_pct"],
        "tau0_kpa": r["tau0_kpa"],
        "flow_mm": r["flow_mm"],
        "n_max": r["n_max"],
        "fck_mpa": r["fck_mpa"],
        "fck_source": r["fck_source"],
        "fck_mae_mpa": r["fck_mae_mpa"],
        **{f"kg_m3_{k}": v for k, v in w.items()},
        **{f"l_m3_{k}": v for k, v in liq.items()},
        "cost_inr_m3": econ["cost_inr_m3"],
        "carbon_kg_co2_m3": econ["carbon_kg_co2_m3"],
    }
