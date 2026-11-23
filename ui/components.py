"""
HTML builders for the UI. Each returns a single-line string (no blank lines, no indentation)
so Streamlit's markdown renderer treats it as raw HTML. All interpolated values are numbers
or labels from our own code, but text is escaped anyway.
"""

from html import escape

from ui import theme

LOGO = (
    '<svg width="46" height="46" viewBox="0 0 38 38" aria-hidden="true" style="flex:none">'
    '<rect x="9" y="6" width="22" height="7" rx="3.5" fill="#9CCBEB"/>'
    '<rect x="5" y="15.5" width="28" height="7" rx="3.5" fill="#7CCBBF"/>'
    '<rect x="11" y="25" width="22" height="7" rx="3.5" fill="#167A6F"/></svg>'
)


def header(ai_label: str, ai_live: bool) -> str:
    state = "live" if ai_live else "off"
    return (
        '<div class="hdr"><div class="brand">' + LOGO +
        '<div><h1>3DCP Mix Designer</h1>'
        '<p>Multilevel MAA particle packing for 3D-printed concrete. Move a slider and the packing, '
        'printability, strength, cost and carbon update together.</p></div></div>'
        f'<span class="chip {state}"><i class="dot"></i>{escape(ai_label)}</span></div>'
    )


def composition(fractions: dict, weights_kg: dict) -> str:
    segs, legend = [], []
    for key, mat in theme.MATERIALS.items():
        frac = fractions[key]
        label = f"{frac * 100:.0f}%" if frac >= 0.07 else ""
        segs.append(f'<span class="seg" style="flex:{max(frac, 0.0001):.4f};background:{mat["mid"]}" '
                    f'title="{escape(mat["label"])} {frac:.1%}">{label}</span>')
        legend.append(
            f'<li><span class="nm"><i style="background:{mat["mid"]}"></i>{escape(mat["label"])}</span>'
            f'<span class="pc">{frac * 100:.1f}%</span>'
            f'<span class="kg">{weights_kg[key]:,.0f} kg/m³</span></li>'
        )
    return (
        '<section class="mix"><div class="mix-head"><h2>What the mix is made of</h2>'
        '<span>Share of solid volume, multilevel MAA result</span></div>'
        '<div class="bead" role="img" aria-label="Solid volume shares by material">' + "".join(segs) + '</div>'
        '<ul class="mix-legend">' + "".join(legend) + '</ul></section>'
    )


def readouts(items: list) -> str:
    """One panel of figures. items: dict(label, value, unit='', note='', status='')."""
    cells = []
    for it in items:
        unit = f'<small>{escape(it["unit"])}</small>' if it.get("unit") else ""
        status = it.get("status", "")
        note = it.get("note", "")
        dot = '<i class="dot"></i>' if status else ""
        cells.append(
            f'<div class="ro"><div class="ro-l">{escape(it["label"])}</div>'
            f'<div class="ro-v">{escape(it["value"])}{unit}</div>'
            f'<div class="ro-n {status}">{dot}{escape(note)}</div></div>'
        )
    return f'<div class="readouts n{len(items)}">' + "".join(cells) + '</div>'


def check(status: str, title: str, detail: str = "") -> str:
    d = f'<span class="d">{escape(detail)}</span>' if detail else ""
    return f'<div class="chk {status}"><i class="dot"></i><div><b>{escape(title)}</b>{d}</div></div>'


def batch_table(headers: list, groups: list, total: list) -> str:
    """groups: [(group_title, [dict(name, color, cells)])]; total: list of cells for the footer row."""
    head = "".join(f'<th class="{"" if i == 0 else "num"}">{escape(h)}</th>' for i, h in enumerate(headers))
    body = []
    for title, rows in groups:
        body.append(f'<tr class="grp"><td colspan="{len(headers)}">{escape(title)}</td></tr>')
        for r in rows:
            sw = f'<i class="sw" style="background:{r["color"]}"></i>' if r.get("color") else '<i class="sw none"></i>'
            cells = "".join(f'<td class="num">{escape(c)}</td>' for c in r["cells"])
            body.append(f'<tr><td>{sw}{escape(r["name"])}</td>{cells}</tr>')
    foot = "".join(f'<td class="num">{escape(c)}</td>' for c in total)
    return (
        f'<table class="bt"><thead><tr>{head}</tr></thead><tbody>' + "".join(body) +
        f'<tr class="tot"><td>Total</td>{foot}</tr></tbody></table>'
    )


def empty_state(text: str) -> str:
    return f'<div class="empty">{escape(text)}</div>'


def copilot_intro() -> str:
    return (
        '<div class="cp-hello"><div class="cp-mark">' + LOGO + '</div>'
        '<div><h3>Ask about this mix</h3>'
        '<p>The Co-Pilot reads your current parameters and results, so it can explain what is holding the mix '
        'back and what to change. Start with one of these, or type your own question below.</p></div></div>'
    )


def copilot_context(row: dict) -> str:
    """Panel listing the live values the Co-Pilot is given with every question."""
    def line(label, value, unit=""):
        u = f'<small>{escape(unit)}</small>' if unit else ""
        return f'<div class="kv"><dt>{escape(label)}</dt><dd>{escape(value)}{u}</dd></div>'

    groups = [
        ("Packing model", [
            line("Powder breakpoint p", f"{row['p']:.3f}"), line("Powder modulus q₁", f"{row['q1']:.3f}"),
            line("Aggregate modulus q₂", f"{row['q2']:.3f}")]),
        ("Mix chemistry", [
            line("Water / binder", f"{row['wb']:.3f}"), line("Superplasticizer", f"{row['sp_pct']:.1f}", "%"),
            line("VMA", f"{row['vma_pct']:.2f}", "%")]),
        ("Printer", [
            line("Nozzle", f"{row['nozzle_mm']:.0f}", "mm"), line("Layer height", f"{row['layer_height_mm']:.0f}", "mm")]),
        ("Results", [
            line("Fit to target R²", f"{row['r2_multilevel']:.3f}"), line("Packing density Φ", f"{row['phi']:.3f}"),
            line("Yield stress", f"{row['tau0_kpa']:.1f}", "kPa"), line("Flow spread", f"{row['flow_mm']:.0f}", "mm"),
            line("Max layers", f"{row['n_max']:,}"), line("Strength", f"{row['fck_mpa']:.0f}", "MPa"),
            line("Cost", f"₹{row['cost_inr_m3']:,.0f}", "/m³"), line("Carbon", f"{row['carbon_kg_co2_m3']:,.0f}", "kg CO₂/m³")]),
    ]
    body = "".join(f'<h4>{escape(t)}</h4><dl>{"".join(ls)}</dl>' for t, ls in groups)
    return (
        '<aside class="cp-ctx"><div class="cp-ctx-head"><h3>What the Co-Pilot sees</h3>'
        '<span>Sent with every question</span></div>' + body + '</aside>'
    )
