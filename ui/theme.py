"""Palette, fonts and Plotly styling shared across the UI."""

from pathlib import Path

import streamlit as st

INK, MUTED, LINE = "#1B2B34", "#5D6E76", "#DCE6E3"
PAPER, PANEL, SIDEBAR = "#F6F9F8", "#FFFFFF", "#EAF3F0"
TEAL, TEAL_MID, TEAL_PALE = "#167A6F", "#7CCBBF", "#D4EFE9"
SLATE = "#8A9AA5"
GAP_FILL, GAP_TEXT = "#FCE3CC", "#9A5A0C"

# One hue per material: pale (table fills), mid (composition strip), deep (chart lines).
MATERIALS = {
    "opc_53":      dict(label="OPC 53 cement",  short="Cement",       pale="#CFE5F5", mid="#9CCBEB", deep="#3A78AE"),
    "fly_ash":     dict(label="Class F fly ash", short="Fly ash",      pale="#E1D9F5", mid="#C3B5EC", deep="#7561BA"),
    "silica_fume": dict(label="Silica fume",    short="Silica fume",  pale="#F8D6DF", mid="#F1A9BC", deep="#C0506F"),
    "fine_sand":   dict(label="Fine sand",      short="Fine sand",    pale="#F7EBB6", mid="#F0D774", deep="#A98410"),
    "medium_sand": dict(label="Medium sand",    short="Medium sand",  pale="#D5EBC6", mid="#A9D68E", deep="#5C8F3F"),
}
LIQUIDS = {"water": "Water", "pce_sp": "PCE superplasticizer", "vma": "VMA"}

FONT = "Instrument Sans, system-ui, sans-serif"
_CSS = Path(__file__).with_name("style.css")


def inject_css() -> None:
    st.markdown(f"<style>{_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def plot_layout(**overrides) -> dict:
    """Shared Plotly layout: white panel, soft grid, muted ticks."""
    axis = dict(gridcolor="#E9F0ED", linecolor=LINE, zeroline=False, tickfont=dict(color=MUTED, size=12),
                title_font=dict(color=MUTED, size=13))
    layout = dict(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        margin=dict(t=28, b=10, l=10, r=32),
        xaxis=dict(axis), yaxis=dict(axis),
        legend=dict(orientation="h", y=-0.22, font=dict(size=12.5, color=INK)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family=FONT, color=INK, size=13)),
    )
    layout.update(overrides)
    return layout


PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
