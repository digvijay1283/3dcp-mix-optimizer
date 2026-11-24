"""AI Co-Pilot (M5): live-state context injection, Gemini streaming, offline fallback."""

from typing import Iterator

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_LABEL = "Gemini 2.0 Flash"

QUICK_PROMPTS = {
    "Diagnose extrudability": "Diagnose whether this mix is extrudable and pumpable. Check flow spread, nozzle choking and yield stress.",
    "Increase buildable layers": "How can I increase the number of buildable layers without hurting extrudability?",
    "Lower cost and carbon": "Suggest changes to lower cost and embodied carbon while keeping the mix printable.",
}

# One line under each preset card, saying what the answer will cover.
QUICK_PROMPT_HINTS = {
    "Diagnose extrudability": "Flow, yield stress and nozzle clogging",
    "Increase buildable layers": "Stiffer mix without losing pumpability",
    "Lower cost and carbon": "Cheaper, greener, still printable",
}


def suggested_prompts(row: dict) -> list:
    """Follow-up questions chosen from the live mix state, most relevant first (max 3)."""
    out = []
    flow = row["flow_mm"]
    if flow < 140:
        out.append("Why is the flow spread below the printable window, and what should I change first?")
    elif flow > 170:
        out.append("How do I bring the flow spread down without making the mix hard to pump?")
    if row["nozzle_mm"] and 2.36 / row["nozzle_mm"] > 0.15:
        out.append("Which nozzle size or top aggregate size avoids clogging for this mix?")
    if row["phi"] > 0.75:
        out.append("Why is the packing density so high, and how far can I trust the layer count?")
    if row["r2_multilevel"] < 0.95:
        out.append("How can I improve the fit between the blended mix and the target curve?")
    out.append("What is the weakest part of this mix right now?")
    return out[:3]


def build_system_prompt(row: dict) -> str:
    return (
        "You are an expert 3D concrete printing (3DCP) mix-design engineer helping a user tune a "
        "Multilevel MAA particle-packing mix. Be concise and concrete; give specific parameter "
        "changes (with direction and rough magnitude) rather than generic advice. Target window: "
        "flow spread 140-170 mm; nozzle clogging risk if D_L/D_nozzle > 0.15 (D_L = 2.36 mm).\n\n"
        "LIVE MIX STATE:\n"
        f"- p={row['p']:.3f}, q1={row['q1']:.3f}, q2={row['q2']:.3f}, W/B={row['wb']:.3f}, "
        f"SP={row['sp_pct']:.1f}%, VMA={row['vma_pct']:.2f}%\n"
        f"- Nozzle={row['nozzle_mm']:.0f} mm, layer height={row['layer_height_mm']:.0f} mm\n"
        f"- Multilevel R2={row['r2_multilevel']:.4f} (single-curve R2={row['r2_single']:.4f})\n"
        f"- Wet packing density={row['phi']:.3f}, yield stress={row['tau0_kpa']:.2f} kPa, "
        f"flow={row['flow_mm']:.1f} mm, max layers={row['n_max']}\n"
        f"- Cost={row['cost_inr_m3']:.0f} INR/m3, carbon={row['carbon_kg_co2_m3']:.0f} kg CO2/m3"
    )


def rule_based_reply(row: dict, question: str) -> str:
    """Offline fallback: inspects live state and returns targeted advice."""
    notes = []
    flow = row["flow_mm"]
    if flow < 140:
        notes.append(
            f"Flow spread is {flow:.0f} mm, below the 140-170 mm window: the mix will be hard to pump. "
            "Raise SP dosage (+0.2-0.5%) or W/B slightly (+0.005-0.01)."
        )
    elif flow > 170:
        notes.append(
            f"Flow spread is {flow:.0f} mm, above the 170 mm window: layers will slump. "
            "Lower W/B (-0.005-0.01) or add VMA (+0.05-0.1%)."
        )
    else:
        notes.append(f"Flow spread {flow:.0f} mm is inside the 140-170 mm printable window.")

    if row["nozzle_mm"] and 2.36 / row["nozzle_mm"] > 0.15:
        notes.append(
            f"Nozzle choking risk: D_L/D_nozzle = {2.36 / row['nozzle_mm']:.3f} > 0.15. "
            "Use a nozzle of at least 16 mm or switch to a finer top-size aggregate."
        )

    if row["r2_multilevel"] < 0.95:
        notes.append(
            f"Multilevel fit R2 = {row['r2_multilevel']:.3f} is modest; try moving p or q2 so the target "
            "better matches what the available materials can deliver."
        )

    q = question.lower()
    if "layer" in q or "build" in q:
        notes.append(
            f"Max layers is {row['n_max']} (yield stress {row['tau0_kpa']:.1f} kPa). Higher silica fume, "
            "VMA, or a lower W/B raise yield stress; check flow stays in-window afterwards."
        )
    if "cost" in q or "carbon" in q:
        notes.append(
            f"Currently {row['cost_inr_m3']:.0f} INR/m3 and {row['carbon_kg_co2_m3']:.0f} kg CO2/m3. "
            "Cement dominates carbon: shifting volume from OPC to fly ash cuts both, while silica fume "
            "dominates cost per kg."
        )
    return "\n\n".join(f"- {n}" for n in notes) + "\n\n_(Offline rule-based Co-Pilot: set GEMINI_API_KEY for full AI answers.)_"


def stream_reply(row: dict, history: list, question: str, api_key: str | None) -> Iterator[str]:
    """Yields reply text chunks. Falls back to rules if no key or the API call fails."""
    if not api_key:
        yield rule_based_reply(row, question)
        return
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])])
            for m in history
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        models_to_try = [GEMINI_MODEL, "gemini-1.5-flash"]
        last_exc = None
        for m_name in models_to_try:
            try:
                stream = client.models.generate_content_stream(
                    model=m_name,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=build_system_prompt(row)),
                )
                got_any = False
                for chunk in stream:
                    if chunk.text:
                        got_any = True
                        yield chunk.text
                if got_any:
                    return
            except Exception as e:
                last_exc = e
                err_str = str(e)
                if "403" in err_str or "API_KEY_INVALID" in err_str or "Forbidden" in err_str:
                    break
                continue
        if last_exc:
            raise last_exc
        yield rule_based_reply(row, question)
    except Exception as exc:  # network/quota/auth failures must not break the demo
        err_str = str(exc)
        if "403" in err_str or "Forbidden" in err_str or "policy" in err_str:
            detail = "API key restricted or forbidden by policy"
        elif "400" in err_str or "API_KEY_INVALID" in err_str:
            detail = "Invalid API key"
        elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            detail = "Quota exceeded"
        else:
            detail = type(exc).__name__
        yield f"_(Gemini unavailable: {detail}. Showing offline analysis.)_\n\n" + rule_based_reply(row, question)
