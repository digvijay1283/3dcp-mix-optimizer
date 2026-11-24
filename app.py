"""3DCP Multilevel MAA Mix Optimization & Engineering Suite (Streamlit UI)."""

import math
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import copilot, optimization, pipeline
from ui import components as ui
from ui import theme

st.set_page_config(page_title="3DCP Mix Designer", page_icon=":material/layers:", layout="wide")
theme.inject_css()

MIXES_CSV = Path(__file__).parent / "data" / "mixes.csv"
AVATAR = Path(__file__).parent / "ui" / "copilot.svg"
MATERIALS = theme.MATERIALS


@st.cache_data
def load_psd():
    return optimization.load_psd_data()


@st.cache_data(show_spinner=False)
def compute(p, q1, q2, q_single, wb, sp_pct, vma_pct, nozzle_mm, layer_height_mm, batch_l):
    params = dict(
        p=p, q1=q1, q2=q2, q_single=q_single, wb=wb, sp_pct=sp_pct, vma_pct=vma_pct,
        nozzle_mm=nozzle_mm, layer_height_mm=layer_height_mm, batch_l=batch_l,
    )
    return pipeline.run_all(load_psd(), params)


def gemini_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


# ---------------------------------------------------------------- M1 sidebar
with st.sidebar:
    st.markdown("### Packing model")
    p = st.slider("Powder breakpoint (p)", 0.400, 0.600, 0.450, 0.005, format="%.3f",
                  help="Target powder share of solids by volume. Sets the height of the flat step across the 90–150 µm gap.")
    q1 = st.slider("Powder modulus (q₁)", 0.010, 0.200, 0.083, 0.005, format="%.3f",
                   help="Distribution modulus for particles up to 90 µm. Lower values mean more fines.")
    q2 = st.slider("Aggregate modulus (q₂)", 0.010, 0.500, 0.255, 0.010, format="%.3f",
                   help="Distribution modulus for particles from 150 µm up.")
    q_single = st.slider("Single-curve modulus (q)", 0.15, 0.35, 0.23, 0.01,
                         help="Modulus of the conventional single-curve MAA target, used for comparison.")
    st.markdown("### Mix chemistry")
    wb = st.slider("Water / binder (W/B)", 0.160, 0.260, 0.200, 0.005, format="%.3f",
                   help="Water mass divided by binder mass (cement, fly ash and silica fume).")
    sp_pct = st.slider("PCE superplasticizer (%)", 0.5, 3.5, 1.5, 0.1, help="Percent of binder mass.")
    vma_pct = st.slider("VMA (%)", 0.00, 0.50, 0.10, 0.05, help="Viscosity-modifying admixture, percent of binder mass.")
    st.markdown("### Printer and batch")
    nozzle_mm = st.slider("Nozzle diameter (mm)", 10, 40, 20,
                          help="Inner nozzle diameter. The largest particle (2.36 mm) should stay under 15% of it.")
    layer_height_mm = st.slider("Layer height (mm)", 5, 20, 10,
                                help="Turns the buildable height into a layer count.")
    batch_l = st.number_input("Lab batch volume (L)", min_value=0.1, value=15.0, step=1.0,
                              help="Sets the quantities and cost in the batch sheet.")

result = compute(p, q1, q2, q_single, wb, sp_pct, vma_pct, nozzle_mm, layer_height_mm, batch_l)
ml, sg = result["opt"]["multilevel"], result["opt"]["single"]
rheo, econ = result["rheo"], result["batch"]["economics"]
b1, bs = result["batch"]["batch_1m3"], result["batch"]["scaled_batch"]
batch_cost = result["batch"]["scaled_cost"]
row = pipeline.summary_row(result)
phi_high = rheo["phi"] > 0.75

# ---------------------------------------------------------------- header, mix strip, headline figures
has_key = bool(gemini_key())
st.markdown(ui.header("Co-Pilot: Gemini" if has_key else "Co-Pilot: offline rules", has_key), unsafe_allow_html=True)
st.markdown(ui.composition(ml["fractions"], b1["weights_kg"]), unsafe_allow_html=True)
st.markdown(ui.readouts([
    dict(label="Fit to target (R²)", value=f"{ml['r2']:.3f}", status="ok" if ml["r2"] >= sg["r2"] else "warn",
         note=f"{ml['r2'] - sg['r2']:+.3f} vs single curve"),
    dict(label="Packing density (Φ)", value=f"{rheo['phi']:.3f}", status="warn" if phi_high else "",
         note="Above typical range" if phi_high else f"{rheo['voids_pct']:.1f}% voids"),
    dict(label="Yield stress (τ₀)", value=f"{rheo['tau0_kpa']:.1f}", unit="kPa", note="Static"),
    dict(label="Max layers", value=f"{rheo['n_max']:,}", status="warn" if phi_high else "",
         note="Likely overstated" if phi_high else f"At {layer_height_mm} mm layers"),
    dict(label="Cost", value=f"₹{econ['cost_inr_m3']:,.0f}", unit="/m³",
         note=f"₹{batch_cost['total_inr']:,.0f} for {batch_l:g} L"),
    dict(label="Embodied carbon", value=f"{econ['carbon_kg_co2_m3']:,.0f}", unit="kg CO₂/m³"),
]), unsafe_allow_html=True)

tab_psd, tab_ml, tab_batch, tab_ai = st.tabs(
    ["PSD curves & diagnostics", "3D Packing surface", "Batching & cost", "AI Co-Pilot"]
)

# ---------------------------------------------------------------- Tab 1: PSD + diagnostics
with tab_psd:
    psd = load_psd()
    sieves = result["opt"]["sieves_um"]

    view = st.segmented_control("X-axis view", ["Gap zoom (30–600 µm)", "Full range (0.05–2,360 µm)"],
                                default="Gap zoom (30–600 µm)", label_visibility="collapsed") or "Gap zoom (30–600 µm)"
    if view.startswith("Gap"):
        xaxis = dict(range=[math.log10(30), math.log10(600)],
                     tickvals=[30, 50, 90, 150, 300, 600], ticktext=["30", "50", "90", "150", "300", "600"])
    else:
        xaxis = dict(range=[math.log10(0.05), math.log10(2600)],
                     tickvals=[0.1, 1, 10, 100, 1000], ticktext=["0.1", "1", "10", "100", "1,000"])

    fig = go.Figure()
    fig.add_vrect(x0=90, x1=150, fillcolor=theme.GAP_FILL, opacity=1, layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=[math.sqrt(90 * 150)], y=[8], mode="text", text=["Gap<br>90–150 µm"],
                             textfont=dict(color=theme.GAP_TEXT, size=13), hoverinfo="skip", showlegend=False))
    for key, mat in MATERIALS.items():
        fig.add_trace(go.Scatter(x=sieves, y=psd["materials"][key]["psd_passing"], name=mat["label"], mode="lines",
                                 line=dict(width=1.6, dash="dot", color=mat["deep"]), opacity=0.9,
                                 hovertemplate=mat["label"] + ": %{y:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=sieves, y=[v * 100 for v in sg["target_curve"]], name="Single-curve target",
                             line=dict(color=theme.SLATE, dash="dash", width=2),
                             hovertemplate="Single-curve target: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=sieves, y=[v * 100 for v in ml["target_curve"]], name="Multilevel target",
                             line=dict(color=theme.TEAL_MID, dash="dash", width=3),
                             hovertemplate="Multilevel target: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=sieves, y=[v * 100 for v in ml["blended_curve"]], name="Blended mix",
                             mode="lines+markers", line=dict(color=theme.TEAL, width=3.5),
                             marker=dict(size=8, color="white", line=dict(color=theme.TEAL, width=2)),
                             hovertemplate="Blended mix: %{y:.1f}%<extra></extra>"))
    fig.update_layout(**theme.plot_layout(height=500, hovermode="closest"))
    fig.update_xaxes(type="log", title="Particle size (µm)", minor=dict(showgrid=False), **xaxis)
    fig.update_yaxes(title="Cumulative passing (%)", range=[0, 102])
    st.plotly_chart(fig, width="stretch", config=theme.PLOT_CONFIG)

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Optimal volume fractions")
        names = [m["short"] for m in MATERIALS.values()]
        bars = go.Figure()
        bars.add_trace(go.Bar(x=names, y=[ml["fractions"][k] for k in MATERIALS], name="Multilevel MAA",
                              marker_color=theme.TEAL, texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False))
        bars.add_trace(go.Bar(x=names, y=[sg["fractions"][k] for k in MATERIALS], name="Single-curve MAA",
                              marker_color="#C5D2D8", texttemplate="%{y:.0%}", textposition="outside", cliponaxis=False))
        bars.update_layout(**theme.plot_layout(height=330, barmode="group", bargap=0.28, bargroupgap=0.06,
                                               legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12.5))))
        bars.update_yaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(bars, width="stretch", config=theme.PLOT_CONFIG)
        st.markdown(
            f'<p class="note">Multilevel fit: RSS {ml["rss"]:.4f}, R² {ml["r2"]:.4f}. '
            f'Single curve: RSS {sg["rss"]:.4f}, R² {sg["r2"]:.4f}.</p>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Printability checks")
        if rheo["flow_in_window"]:
            st.markdown(ui.check("ok", f"Flow spread {rheo['flow_mm']:.1f} mm", "Inside the 140–170 mm window."), unsafe_allow_html=True)
        else:
            side = "below" if rheo["flow_mm"] < 140 else "above"
            st.markdown(ui.check("warn", f"Flow spread {rheo['flow_mm']:.1f} mm",
                                 f"Outside the 140–170 mm window ({side})."), unsafe_allow_html=True)
        ratio = 2.36 / nozzle_mm
        if rheo["nozzle_choke_warning"]:
            st.markdown(ui.check("bad", "Nozzle may clog", f"Largest particle is {ratio:.0%} of the nozzle; keep it at or below 15%."), unsafe_allow_html=True)
        else:
            st.markdown(ui.check("ok", "Nozzle is clear", f"Largest particle is {ratio:.1%} of the nozzle (limit 15%)."), unsafe_allow_html=True)
        lo, hi = optimization.FRACTION_BOUNDS_LO, optimization.FRACTION_BOUNDS_HI
        at_bound = [
            MATERIALS[k]["label"] for k, a, b in zip(optimization.MATERIAL_KEYS, lo, hi)
            if abs(ml["fractions"][k] - a) < 1e-3 or abs(ml["fractions"][k] - b) < 1e-3
        ]
        if at_bound:
            st.markdown(ui.check("warn", "Solver hit a limit", "Fraction pinned at its bound for " + ", ".join(at_bound) + "."), unsafe_allow_html=True)
        if phi_high:
            st.markdown(ui.check("warn", "Packing density looks high",
                                 "Φ above about 0.75 is higher than typical literature values, so the layer count is "
                                 "likely overstated. This comes from the regression, not from a calculation error."),
                        unsafe_allow_html=True)
        st.write("")
        st.markdown(ui.readouts([
            dict(label="Voids", value=f"{rheo['voids_pct']:.1f}", unit="%"),
            dict(label="Max buildable height", value=f"{rheo['h_max_m']:.2f}", unit="m"),
        ]), unsafe_allow_html=True)

# ---------------------------------------------------------------- Tab 2: 3D Packing surface
with tab_ml:
    p_axis, q1_axis, z = pipeline.phi_surface(q2)
    surf = go.Figure(go.Surface(
        x=p_axis, y=q1_axis, z=z, opacity=0.96,
        colorscale=[[0, "#E7F5F1"], [0.35, "#A7DACF"], [0.65, "#3FA796"], [1, "#1C5F73"]],
        colorbar=dict(title="Φ", thickness=12, len=0.7, outlinewidth=0, tickformat=".2f", x=0.93, xanchor="left"),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        hovertemplate="p %{x:.3f}<br>q₁ %{y:.3f}<br>Φ %{z:.3f}<extra></extra>",
    ))
    surf.add_trace(go.Scatter3d(x=[p], y=[q1], z=[rheo["phi"]], mode="markers", name="Current mix",
                                marker=dict(size=10, color="#C0506F", line=dict(color="white", width=2)),
                                hovertemplate="Current mix<br>Φ %{z:.3f}<extra></extra>"))
    axis3d = dict(backgroundcolor=theme.PAPER, gridcolor=theme.LINE, showbackground=True, zerolinecolor=theme.LINE)
    surf.update_layout(**theme.plot_layout(
        height=540, showlegend=False, margin=dict(t=10, b=0, l=0, r=64),
        scene=dict(xaxis=dict(title="Powder breakpoint p", **axis3d), yaxis=dict(title="Powder modulus q₁", **axis3d),
                   zaxis=dict(title="Φ", **axis3d), aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.55),
                   camera=dict(eye=dict(x=1.45, y=-1.45, z=0.85))),
    ))
    st.plotly_chart(surf, width="stretch", config=theme.PLOT_CONFIG)
    st.write("")

    st.markdown(ui.readouts([
        dict(label="Flow spread", value=f"{rheo['flow_mm']:.1f}", unit="mm",
             status="ok" if rheo["flow_in_window"] else "warn",
             note="In window" if rheo["flow_in_window"] else "Outside 140–170 mm"),
        dict(label="Static yield stress", value=f"{rheo['tau0_kpa']:.1f}", unit="kPa"),
        dict(label="Packing density (Φ)", value=f"{rheo['phi']:.3f}",
             note=f"{rheo['voids_pct']:.1f}% voids"),
    ]), unsafe_allow_html=True)

    st.markdown(
        '<p class="note">The 3D surface shows wet packing density Φ over powder breakpoint p and powder modulus q₁ '
        'at the current aggregate modulus q₂.</p>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------- Tab 3: batching
with tab_batch:
    cost_col = f"Cost, {batch_l:g} L batch"
    solids, liquids, sheet_rows = [], [], []
    for k, mat in MATERIALS.items():
        cells = [f"{ml['fractions'][k]:.1%}", f"{b1['weights_kg'][k]:.1f} kg", f"{bs['weights_kg'][k]:.3f} kg",
                 f"₹{batch_cost['per_component_inr'][k]:,.2f}"]
        solids.append(dict(name=mat["label"], color=mat["mid"], cells=cells))
        sheet_rows.append([mat["label"], *cells])
    for k, label in theme.LIQUIDS.items():
        cells = ["—", f"{b1['liquids_l'][k]:.2f} L", f"{bs['liquids_ml'][k]:.1f} mL",
                 f"₹{batch_cost['per_component_inr'][k]:,.2f}"]
        liquids.append(dict(name=label, cells=cells))
        sheet_rows.append([label, *cells])
    headers = ["Component", "Share of solids", "1 m³", f"{batch_l:g} L batch", cost_col]
    total = ["100%", f"{b1['wet_density_kg_m3']:,.0f} kg", f"{b1['wet_density_kg_m3'] * batch_l / 1000:.3f} kg",
             f"₹{batch_cost['total_inr']:,.2f}"]
    st.markdown(ui.batch_table(headers, [("Solids", solids), ("Liquids and admixtures", liquids)], total),
                unsafe_allow_html=True)
    st.markdown('<p class="note">Absolute-volume method with 2% entrapped air. The total row gives the wet mass '
                'of the mix.</p>', unsafe_allow_html=True)

    st.markdown(ui.readouts([
        dict(label="Wet density", value=f"{b1['wet_density_kg_m3']:,.0f}", unit="kg/m³"),
        dict(label="Cost", value=f"₹{econ['cost_inr_m3']:,.0f}", unit="/m³"),
        dict(label=f"Cost of {batch_l:g} L batch", value=f"₹{batch_cost['total_inr']:,.2f}"),
        dict(label="Embodied carbon", value=f"{econ['carbon_kg_co2_m3']:,.0f}", unit="kg CO₂/m³"),
    ]), unsafe_allow_html=True)

    sheet = pd.DataFrame(sheet_rows, columns=headers)
    a, b, _ = st.columns([1.35, 1, 3.4])
    a.download_button("Download batch sheet", sheet.to_csv(index=False), "batch_sheet.csv", "text/csv")
    if b.button("Log this mix"):
        MIXES_CSV.parent.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(MIXES_CSV, mode="a", header=not MIXES_CSV.exists(), index=False)
        st.toast("Mix logged to data/mixes.csv")
    if MIXES_CSV.exists():
        with st.expander("Logged mixes"):
            st.dataframe(pd.read_csv(MIXES_CSV), width="stretch")

# ---------------------------------------------------------------- Tab 4: AI co-pilot
with tab_ai:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages = st.session_state.messages
    mix_key = (p, q1, q2, q_single, wb, sp_pct, vma_pct, nozzle_mm, layer_height_mm, batch_l)
    answerer = copilot.GEMINI_LABEL if has_key else "offline rules"

    chat_col, ctx_col = st.columns([2.3, 1], gap="large")
    pending = None

    with chat_col:
        if not messages:
            st.markdown(ui.copilot_intro(), unsafe_allow_html=True)
            with st.container(key="prompt_cards"):
                for col, (label, prompt) in zip(st.columns(3), copilot.QUICK_PROMPTS.items()):
                    if col.button(f"**{label}**\n\n{copilot.QUICK_PROMPT_HINTS[label]}", key=f"card_{label}", width="stretch"):
                        pending = prompt
        else:
            head, clear = st.columns([5, 1])
            head.markdown(f'<div class="cp-head"><span class="who">Answering with {answerer}</span></div>',
                          unsafe_allow_html=True)
            if clear.button("Clear chat", width="stretch"):
                st.session_state.messages = []
                st.rerun()

        for m in messages:
            if m["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(m["content"])
            else:
                with st.chat_message("assistant", avatar=str(AVATAR)):
                    st.markdown(m["content"])
                stale = '<span class="stale">Based on an earlier mix</span>' if m.get("mix_key") != mix_key else ""
                st.markdown(f'<div class="cp-meta"><span>{m.get("by", "")}</span>{stale}</div>', unsafe_allow_html=True)

        if messages:
            st.markdown('<div class="cp-next">Ask next</div>', unsafe_allow_html=True)
            with st.container(key="followups"):
                for i, prompt in enumerate(copilot.suggested_prompts(row)):
                    if st.button(prompt, key=f"follow_{len(messages)}_{i}"):
                        pending = prompt

        question = st.chat_input("Ask about this mix…") or pending
        if question:
            history = list(messages)
            messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant", avatar=str(AVATAR)):
                reply = st.write_stream(copilot.stream_reply(row, history, question, gemini_key()))
            messages.append({"role": "assistant", "content": reply, "mix_key": mix_key, "by": answerer})
            st.rerun()

    with ctx_col:
        st.markdown(ui.copilot_context(row), unsafe_allow_html=True)
        if not has_key:
            st.markdown('<p class="note">No GEMINI_API_KEY found, so answers come from the offline rule-based Co-Pilot.</p>',
                        unsafe_allow_html=True)
