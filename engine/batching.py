"""
Industrial batching, cost, and carbon analytics (M4).
Absolute volume method: converts solid volume fractions into 1 m^3 weights,
scales to a custom lab batch size, and computes cost/carbon totals.
"""

MATERIAL_KEYS = ["opc_53", "fly_ash", "silica_fume", "fine_sand", "medium_sand"]
ENTRAPPED_AIR_FRACTION = 0.02


def compute_1m3_batch(psd_data: dict, fractions: dict, wb: float, sp_pct: float, vma_pct: float) -> dict:
    """
    Absolute volume method for 1 m^3 of concrete, assuming 2% entrapped air.

    fractions: dict of material_key -> volume fraction of the SOLIDS portion
               (sums to 1.0), as returned by the optimization engine.

    Returns weights in kg for solids, litres for liquids.
    """
    materials = psd_data["materials"]
    liquids = psd_data["liquid_admixtures"]

    available_volume_m3 = 1.0 - ENTRAPPED_AIR_FRACTION

    # Water/SP/VMA are dosed on binder mass, so their volume scales with the solids
    # volume. Solve V_solids + V_liquids = available volume (absolute volume closure).
    binder_mass_per_solid_m3 = sum(
        fractions[k] * materials[k]["density_kg_m3"] for k in ("opc_53", "fly_ash", "silica_fume")
    )
    liquid_m3_per_binder_kg = (
        wb / liquids["water"]["density_kg_m3"]
        + (sp_pct / 100.0) / liquids["pce_sp"]["density_kg_m3"]
        + (vma_pct / 100.0) / liquids["vma"]["density_kg_m3"]
    )
    solid_volume_m3 = available_volume_m3 / (1.0 + binder_mass_per_solid_m3 * liquid_m3_per_binder_kg)

    weights_kg = {}
    for key in MATERIAL_KEYS:
        vol_m3 = solid_volume_m3 * fractions[key]
        weights_kg[key] = vol_m3 * materials[key]["density_kg_m3"]

    # Estimate binder mass (kg) to size water/admixture demand
    binder_mass_kg = sum(weights_kg[k] for k in ["opc_53", "fly_ash", "silica_fume"])

    water_mass_kg = wb * binder_mass_kg
    sp_mass_kg = (sp_pct / 100.0) * binder_mass_kg
    vma_mass_kg = (vma_pct / 100.0) * binder_mass_kg

    # volume_m3 = mass_kg / density_kg_m3; convert to litres (1 m3 = 1000 L)
    liquids_l = {
        "water": (water_mass_kg / liquids["water"]["density_kg_m3"]) * 1000.0,
        "pce_sp": (sp_mass_kg / liquids["pce_sp"]["density_kg_m3"]) * 1000.0,
        "vma": (vma_mass_kg / liquids["vma"]["density_kg_m3"]) * 1000.0,
    }

    liquid_mass_kg = water_mass_kg + sp_mass_kg + vma_mass_kg
    wet_density = sum(weights_kg.values()) + liquid_mass_kg  # kg in 1 m^3

    return {
        "weights_kg": weights_kg,
        "liquids_l": liquids_l,
        "binder_mass_kg": binder_mass_kg,
        "wet_density_kg_m3": wet_density,
    }


def scale_batch(batch_1m3: dict, batch_volume_l: float) -> dict:
    """Scales a 1 m^3 batch down to a custom lab batch volume (litres).
    Solids returned in kg, liquids returned in mL."""
    scale = batch_volume_l / 1000.0  # 1 m^3 = 1000 L

    scaled_weights_kg = {k: v * scale for k, v in batch_1m3["weights_kg"].items()}
    scaled_liquids_ml = {k: v * scale * 1000.0 for k, v in batch_1m3["liquids_l"].items()}

    return {
        "weights_kg": scaled_weights_kg,
        "liquids_ml": scaled_liquids_ml,
    }


def compute_cost_and_carbon(psd_data: dict, batch_1m3: dict) -> dict:
    """Total cost (INR/m^3) and embodied carbon (kg CO2/m^3) for 1 m^3."""
    materials = psd_data["materials"]
    liquids = psd_data["liquid_admixtures"]

    cost = 0.0
    carbon = 0.0

    for key, weight_kg in batch_1m3["weights_kg"].items():
        cost += weight_kg * materials[key]["cost_inr_kg"]
        carbon += weight_kg * materials[key]["co2_factor"]

    for key, vol_l in batch_1m3["liquids_l"].items():
        cost += vol_l * liquids[key]["cost_inr_l"]
        carbon += (vol_l * liquids[key]["density_kg_m3"] / 1000.0) * liquids[key]["co2_factor"]

    return {
        "cost_inr_m3": cost,
        "carbon_kg_co2_m3": carbon,
    }


def scaled_batch_cost(psd_data: dict, scaled: dict) -> dict:
    """Cost (INR) of each component and the total for the scaled lab batch."""
    materials = psd_data["materials"]
    liquids = psd_data["liquid_admixtures"]

    per_component = {k: w_kg * materials[k]["cost_inr_kg"] for k, w_kg in scaled["weights_kg"].items()}
    per_component.update(
        {k: (ml / 1000.0) * liquids[k]["cost_inr_l"] for k, ml in scaled["liquids_ml"].items()}
    )
    return {"per_component_inr": per_component, "total_inr": sum(per_component.values())}


def run_batching_suite(psd_data: dict, fractions: dict, wb: float, sp_pct: float, vma_pct: float, batch_volume_l: float) -> dict:
    """Runs the full M4 batching pipeline: 1 m^3 -> scaled batch -> cost/carbon."""
    batch_1m3 = compute_1m3_batch(psd_data, fractions, wb, sp_pct, vma_pct)
    scaled = scale_batch(batch_1m3, batch_volume_l)
    economics = compute_cost_and_carbon(psd_data, batch_1m3)
    scaled_cost = scaled_batch_cost(psd_data, scaled)

    return {
        "batch_1m3": batch_1m3,
        "scaled_batch": scaled,
        "economics": economics,
        "scaled_cost": scaled_cost,
    }
