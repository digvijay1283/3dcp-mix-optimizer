# Architecture Document
## Project: AI-Powered 3DCP Multilevel MAA Mix Optimization & Engineering Suite

Stack: **Streamlit (Python), single-file application**

> **Note on scope change:** The PRD (`prd.md`) specifies React + Tailwind + Node.js + Firebase. This architecture documents a deliberate, time-boxed pivot to a single-file Streamlit app for a 3-day delivery deadline. The pivot trades the specified frontend framework, hosted database, and deployment topology for speed; it does **not** change the engineering math (M2–M4 formulas, the MAA solvers, the materials matrix) or the AI Co-Pilot's behavior (M5) — those are implemented identically in Python. See Section 6 for what is gained and what is given up.

---

## 1. System Overview

Everything runs in one Python process, in one file (`app.py`), launched with `streamlit run app.py`. There is no client/server split, no REST API, no separate database service — Streamlit's script-rerun model (the whole script re-executes top-to-bottom on every widget interaction, with cheap steps cached) replaces the React render loop, and the Python process itself does all computation that would otherwise have been split across a Node backend and a Web Worker.

```
┌──────────────────────────────────────────────────────────────┐
│                      streamlit run app.py                      │
│                                                                  │
│  Sidebar Widgets (M1)                                           │
│  st.slider(p, q1, q2, q_single, W/B, SP%, VMA%, nozzle, layer_h)│
│  st.number_input(batch volume)                                  │
│         │                                                        │
│         ▼  (Streamlit reruns script on every widget change)     │
│  @st.cache_data  psd_data (materials matrix, loaded once)        │
│         │                                                        │
│         ▼                                                        │
│  Optimization Engine (M2) — scipy.optimize                       │
│  - single-curve MAA least-squares fit                            │
│  - multilevel piecewise MAA least-squares fit                    │
│  - returns v_OPC, v_FA, v_SF, v_FS, v_MS, RSS, R²                 │
│         │                                                        │
│         ▼                                                        │
│  Rheology & Diagnostics (M3) — plain numpy functions              │
│  - Φ, Voids%, τ0, Flow spread, Hmax, Nmax, nozzle choke check      │
│  - fck prediction (joblib-loaded sklearn pipeline OR fitted       │
│    formula if no trained model exists)                            │
│         │                                                        │
│         ▼                                                        │
│  Batching, Cost & Carbon (M4) — plain numpy/pandas                 │
│  - absolute volume method → 1 m³ weights → scale to batch volume   │
│  - cost (INR/m³), embodied carbon (kg CO₂/m³)                      │
│         │                                                        │
│         ▼                                                        │
│  UI Render — st.tabs() for the 4 workspace tabs                    │
│  - Tab 1: Plotly semi-log PSD curve + diagnostics                  │
│  - Tab 2: Plotly 3D surface + ML prediction display                │
│  - Tab 3: pandas DataFrame batch sheet (st.dataframe)               │
│  - Tab 4: AI Co-Pilot chat (st.chat_message / st.chat_input)        │
│         │                                                        │
│         ▼                                                        │
│  "Log Mix" button → append row to local mixes.csv (pandas)         │
│                                                                    │
│  AI Co-Pilot: google-generativeai SDK call with live state         │
│  injected into the prompt, OR a rule-based fallback if no           │
│  GEMINI_API_KEY is set (keeps the demo working offline)             │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Why Single-File Streamlit Fits This Project

- **M1–M4 are pure, deterministic math** over a small fixed matrix (16 sieves × 5 materials). Streamlit's rerun-on-interaction model is sufficient — there's no need for a persistent server process reacting to events, since every slider move is already a full script rerun with cached heavy steps.
- **`scipy.optimize`** is a better fit for the MAA least-squares solvers than hand-rolled JavaScript — this is the one area where the Python pivot is an upgrade, not just a shortcut.
- **No auth, no multi-user concerns**: this is a single-operator engineering tool (a student/researcher running their own session), not a multi-tenant web product, so Firebase Auth and Firestore security rules solve a problem this project doesn't have at the demo stage.
- **Deployment is one command locally**, and free-tier hosting (Streamlit Community Cloud) needs no separate frontend/backend deploy pipeline.

---

## 3. File Structure

A single file is viable for a 3-day build, but splitting the pure-math engine out keeps it testable and keeps `app.py` readable:

```
/app.py                 ← Streamlit UI: sidebar, tabs, chat, button wiring
/engine
  psd_data.json           (master materials matrix, from PRD Section 5)
  optimization.py         (single-curve + multilevel MAA solvers, scipy.optimize)
  rheology.py              (Φ, τ0, Flow, Hmax, Nmax, nozzle check, fck)
  batching.py               (absolute volume method, scaling, cost, carbon)
/model
  strength_pipeline.joblib  (trained sklearn pipeline, if one exists — see Section 5)
/data
  mixes.csv                  (append-only log, created on first "Log Mix" click)
/.streamlit
  secrets.toml                 (GEMINI_API_KEY — gitignored)
requirements.txt
```

`engine/` functions are plain Python with no Streamlit imports, so they can be unit-tested (`pytest`) independently of the UI — this is where correctness risk actually lives (see the earlier difficulty rating: the solver, not the UI, is the hard part).

---

## 4. Module-by-Module Mapping

### 4.1 M1 — Parameter Control Panel
`st.sidebar` with `st.slider()` for every PRD-specified range/step/default (p, q1, q2, q_single, W/B, SP%, VMA%, nozzle, layer height) and `st.number_input()` for batch volume. Streamlit widgets return the current value directly — no separate state store needed; the script rerun *is* the state update.

### 4.2 M2 — PSD & Dual Optimization Engine
- `psd_data.json` loaded once via `@st.cache_data` (avoids re-parsing on every rerun).
- `optimization.py`: two `scipy.optimize.least_squares` (or `curve_fit`) calls — one fitting the single Fuller-type target curve, one fitting the piecewise multilevel target with the flat gap step at `p` — both solving for volume fractions of the 5 materials, returning RSS and R².
- Plotly semi-log line chart: raw material curves, both target curves, actual blended curve, shaded 90–150 µm gap band (`fig.add_vrect`).

### 4.3 M3 — Rheology, Buildability & ML Predictor
- `rheology.py` implements the PRD's exact closed-form formulas for Φ, τ0, Flow, Hmax, Nmax, and the nozzle choke check (`D_L/D_nozzle > 0.15`) — direct, literal transcription, unit-checked (mm vs m, kPa vs Pa).
- **M3.3 ML predictor — the one open dependency (see Section 5 below).**

### 4.4 M4 — Batching, Cost & Carbon
`batching.py`: absolute volume method assuming 2% entrapped air → 1 m³ weights (kg solids, L liquids) → scaled to `V_batch` → cost (INR/m³) and carbon (kg CO₂/m³) using the PRD's unit rates and emission factors. Rendered as a `pandas.DataFrame` via `st.dataframe` or `st.table` for the batch sheet.

**M4.4 Firestore logging → replaced with local CSV logging.** A "Log Mix" button appends the current parameters/results as a row to `data/mixes.csv` via `pandas.DataFrame.to_csv(mode="append")`. This is the most direct simplification: same *intent* (persist a record of this mix), trivial implementation, but it means logged mixes live only on the machine running the app — no cross-session or cross-device history, and on most free hosting tiers the filesystem is ephemeral (resets on redeploy/restart).

### 4.5 M5 — AI Engineering Co-Pilot
- `google-generativeai` SDK called directly from the Streamlit process (no proxy needed — this is a single-user local/demo tool, not a multi-tenant web app where hiding the key from "the client" matters in the same way). The API key is read from `.streamlit/secrets.toml` via `st.secrets["GEMINI_API_KEY"]`, which is still kept out of source control.
- M5.1 context injection: build a system prompt string from the current live values (q1, q2, p, W/B, R², Φ, τ0, Flow, Nmax, fck, cost, carbon) on every chat turn — straightforward since everything is already in local Python variables from the same script run.
- M5.2 streaming UI: `st.chat_message` / `st.chat_input` plus `st.write_stream` for token streaming; quick-prompt chips as `st.button` row above the chat input.
- **Offline fallback**: if `GEMINI_API_KEY` isn't set, fall back to a small rule-based responder (`if tau0 too low: ... elif flow out of 140-170 window: ...`) so the Co-Pilot tab still responds meaningfully without network/API access — useful for a live demo with unreliable wifi.

---

## 5. Open Dependency: the ML Strength Predictor (M3.3)

This is flagged separately because it's the one piece that isn't pure transcription of a given formula — the PRD says "trained model pipeline" but doesn't specify the training data or algorithm.

- **If a trained model already exists** (e.g. your AI/ML teammate has a `.pkl`/joblib sklearn pipeline from prior work): load it with `joblib.load()` in `rheology.py` and call `.predict([[phi, wb, v_sf, p, q1]])`. This is near-zero additional work.
- **If no trained model exists yet**: for a 3-day deadline, do **not** start an ML data-collection-and-training project. Substitute a simple, clearly-labeled empirical regression (e.g. a published strength-vs-packing-density correlation, or a quick linear/polynomial fit against Φ and W/B alone) and label it in the UI as an "indicative estimate," noting the full ML pipeline as future work. This keeps the demo honest without blocking on a multi-day ML task.

---

## 6. What Changes vs. the Original React/Node/Firebase Architecture

| Aspect | Original (React/Node/Firebase) | This pivot (Streamlit) |
|---|---|---|
| Frontend framework | React + Tailwind, specified color system | Streamlit's built-in theming (close approximation, not pixel-identical to the PRD's Indigo/Amber/Slate spec) |
| Optimization engine | JS in a Web Worker | Python, `scipy.optimize` (arguably better suited to the math) |
| Backend/API | Node/Express, 4 REST endpoints | None — same process, same script |
| Auth | Firebase Auth | None — single-user local tool |
| Persistence | Firestore, per-user documents | Local `mixes.csv`, no per-user separation |
| AI Co-Pilot | Node proxies Gemini, key server-side | Called directly from the Streamlit process, key in `st.secrets` |
| Deployment | Separate frontend/backend deploys | One command locally; Streamlit Community Cloud for free hosting |
| Multi-user support | Yes (by design) | No — not needed for a lab/demo tool, but a real limitation if this needs to serve multiple concurrent users later |

**Bottom line:** the pivot gives up the parts of the original architecture that solved problems this project doesn't currently have (multi-tenant auth, a hosted database, a decoupled API) and keeps everything that actually required engineering care (the solvers, the formulas, the AI context injection). If this later needs to support multiple simultaneous users with persistent accounts, revisit Firebase/Firestore then — don't build it now against a deadline.

---

## 7. Project Implementation Flow (Build Order)

Same dependency order as before — the math must be right before anything downstream is trustworthy — compressed for a single-file build:

### Phase 1 — Engine (Day 1 morning)
Build `engine/optimization.py` and `engine/rheology.py` and `engine/batching.py` as plain Python, tested from a throwaway script or `pytest`, *before* touching Streamlit at all. Validate against the PRD's default slider values.

### Phase 2 — Streamlit UI wiring (Day 1 afternoon – Day 2)
Build `app.py`: sidebar sliders → call into `engine/` → top metric row → the 4 tabs (PSD chart, 3D surface/ML, batch sheet, chat placeholder). Use mock/fixed `fck` until Section 5's ML question is resolved.

### Phase 3 — Logging + ML resolution (Day 2)
Wire the "Log Mix" → CSV button. Resolve M3.3 per Section 5 (load real model if it exists, else swap in the labeled empirical estimate).

### Phase 4 — AI Co-Pilot (Day 2 evening – Day 3)
Wire `google-generativeai`, context injection, streaming chat, quick-prompt chips, and the offline rule-based fallback. Test last, against already-correct numbers from Phases 1–3.

---

## 7a. Known Quirk: Φ Formula and Nmax at Default Values

At the PRD's default slider values (`p=0.450, q1=0.083, q2=0.255`), the M3.1 wet packing density regression yields **Φ ≈ 0.81**, which is higher than typical literature values for 3DCP wet packing density (commonly ~0.60–0.75). This cascades through `H_max`/`N_max` and produces an implausibly high buildable layer count (~400+ layers) at those exact defaults.

This is **not a transcription bug** — the formula in `engine/rheology.py::wet_packing_density` matches PRD Section M3.1 exactly (verified by direct substitution). The formula is shipped as given. Two plausible explanations, neither resolved here: (a) the regression's valid/calibrated input range may be narrower than the full slider range the PRD defines, or (b) this is simply what the source regression (Yin et al., 2023) produces at these inputs and the real buildability limit comes from a different constraint not modeled here. If this number is shown during a demo or report, note it as a known characteristic of the regression rather than a verified physical prediction, and flag it to your guide if asked. Do not silently alter the PRD-given formula to "fix" the output.

---

## 8. Key Decisions (Summary)

1. **Single Python process, single file** — no network boundary anywhere in the app; eliminates the entire class of frontend/backend integration bugs the original architecture had to manage.
2. **`scipy.optimize` for the MAA solvers** — same math as originally planned, better-suited tooling.
3. **Local CSV instead of Firestore** — same intent (persist mix records), trivial implementation, explicitly no multi-user/multi-device guarantee.
4. **Gemini called in-process with an offline rule-based fallback** — keeps the AI Co-Pilot demoable without depending on live internet/API access during a presentation.
5. **ML strength predictor is the one genuinely open item** — resolve by checking whether a trained pipeline already exists; if not, substitute a labeled empirical estimate rather than scoping in an ML project under a 3-day deadline.
