# Product Requirements Document (PRD)
## Project Name: AI-Powered 3D Concrete Printing (3DCP) Multilevel MAA Mix Optimization & Engineering Suite

---

## 1. Project Overview & Business Logic
In 3D Concrete Printing (3DCP), fresh concrete must be pumpable through a hose, extrudable through a narrow nozzle ($10\text{–}40\text{ mm}$), and possess immediate static yield stress (thixotropy) to support stacked layers without plastic collapse or formwork.

Conventional mix designs suffer from a severe **Gap Gradation Problem** between cementitious binders ($D \le 90\ \mu\text{m}$) and fine aggregates ($D \ge 150\ \mu\text{m}$). This web application digitizes and automates the **Multilevel Modified Andreasen & Andersen (MAA) Particle Packing Model** (derived by *Yin et al., 2023*), optimizing solid volume fractions, calculating fresh rheology and 28-day mechanical properties, generating $1\text{ m}^3$ and custom lab batch sheets, tracking material cost & carbon footprint, and providing a context-aware AI Co-Pilot powered by the Gemini API.

---

## 2. User Roles & Persona Objectives
* **Civil Engineering Student / Researcher:** Wants to input particle size distribution (PSD) data, run least-squares optimizations, plot semi-logarithmic PSD curves, and export verified batch sheets for lab mixing.
* **3DCP Machine Operator / Engineer:** Wants to check nozzle clogging risks ($D_L \le 0.15 D_{\text{nozzle}}$), predict workability spread ($140\text{–}170\text{ mm}$ window), calculate maximum buildable layers before bottom-layer buckling, and consult the AI Co-Pilot for real-time mix troubleshooting.

---

## 3. Detailed Functional Modules & Feature Specifications

### Module 1: Real-Time Parameter Control Panel
* **M1.1 Sliders & Input Widgets:**
  * Powder Volume Breakpoint ($p$): Range `0.400` to `0.600`, step `0.005`, default `0.450`.
  * Powder Distribution Modulus ($q_1$): Range `0.010` to `0.200`, step `0.005`, default `0.083`.
  * Aggregate Distribution Modulus ($q_2$): Range `0.010` to `0.500`, step `0.010`, default `0.255`.
  * Baseline Single-MAA Modulus ($q_{\text{single}}$): Range `0.15` to `0.35`, step `0.01`, default `0.23`.
  * Water-to-Binder Mass Ratio ($W/B$): Range `0.160` to `0.260`, step `0.005`, default `0.200`.
  * PCE Superplasticizer Dosage ($SP\%$): Range `0.5%` to `3.5%`, step `0.1%`, default `1.5%`.
  * Viscosity Modifying Admixture ($VMA\%$): Range `0.0%` to `0.5%`, step `0.05%`, default `0.10%`.
  * Nozzle Diameter ($D_{\text{nozzle}}$): Range `10` to `40 mm`, default `20 mm`.
  * Layer Height ($h_{\text{layer}}$): Range `5` to `20 mm`, default `10 mm`.
  * Custom Lab Batch Volume ($V_{\text{batch}}$): Number input, default `15.0 Litres`.

### Module 2: Particle Size Distribution (PSD) & Dual Optimization Engine
* **M2.1 Matrix Processing:** Embedded master PSD matrix containing 16 sieve sizes ($0.05\ \mu\text{m}$ to $2360\ \mu\text{m}$) across 5 raw solid materials: OPC 53 Cement, Class F Fly Ash, Silica Fume, Fine Sand ($150\text{–}600\ \mu\text{m}$), and Medium Sand ($600\text{–}2360\ \mu\text{m}$).
* **M2.2 Dual Optimization Execution:** Executes both conventional Single-Curve MAA and Piecewise Multilevel MAA solvers concurrently on input change, returning:
  * Optimal volume fractions ($v_{\text{OPC}}, v_{\text{FA}}, v_{\text{SF}}, v_{\text{FS}}, v_{\text{MS}}$).
  * Residual Sum of Squares ($RSS$) and Coefficient of Determination ($R^2$).
* **M2.3 Visualization:** Renders an interactive Chart.js / Recharts semi-logarithmic chart showing raw material curves, single target curve, multilevel target curve with flat gap step at $p$, actual blended mix curve, and highlighted $90\text{–}150\ \mu\text{m}$ gap zone.

### Module 3: Rheology, Buildability & Machine Learning Property Predictor
* **M3.1 Density & Void Computation:**
  * Wet Packing Density ($\Phi$) via quadratic response surface regression:
    $$\Phi = 0.4707 + 0.0319 q_1 + 1.5136 p + 0.6182 q_1 p - 1.8392 q_1^2 - 0.0125 q_2^2 - 1.7506 p^2$$
  * Voids percentage: $\text{Voids}\% = (1 - \Phi) \times 100$.
* **M3.2 Printability & Hardware Diagnostics:**
  * Static Yield Stress ($\tau_0$ in kPa):
    $$\tau_0 = 0.42 \times \left(1 - \frac{\Phi}{0.865}\right)^{-1.6} \times \left(\frac{0.20}{W/B}\right)^{1.7} \times \left(1 + 3.5 v_{\text{SF}} + 1.8 \text{VMA}\%\right)$$
  * Workability Flow Spread (mm):
    $$\text{Flow} = 155 + 650(W/B - 0.20) + 14(SP\% - 1.5) - 180(v_{\text{SF}} - 0.05) - 25(\text{VMA}\%)$$
  * Maximum Buildable Height ($H_{\max}$) and Layer Count ($N_{\max}$):
    $$H_{\max} = \frac{\sqrt{3} \cdot \tau_0 \cdot 1000}{\rho_{\text{wet}} \cdot 9.81}, \qquad N_{\max} = \left\lfloor \frac{H_{\max} \times 1000}{h_{\text{layer}}} \right\rfloor$$
  * Nozzle Choking Check: Flag warning if $D_L / D_{\text{nozzle}} > 0.15$ ($D_L = 2.36\text{ mm}$).
* **M3.3 ML Compressive Strength Predictor:** Predicts 28-day Compressive Strength ($f_{ck}$ in MPa) based on trained model pipeline using $\Phi$, $W/B$, $v_{\text{SF}}$, $p$, and $q_1$.

### Module 4: Industrial Batching, Cost & Carbon Analytics
* **M4.1 Absolute Volume Batch Conversion:** Converts optimal solid volume fractions into $1\text{ m}^3$ weights (kg for solids, Litres for liquids) assuming $2\%$ entrapped air.
* **M4.2 Scaling:** Scales $1\text{ m}^3$ values to user's custom lab batch size ($V_{\text{batch}}$ in Litres) with solid units in kg and liquids in mL.
* **M4.3 Economics & Sustainability:** Calculates total cost ($\text{INR/m}^3$) and embodied carbon ($\text{kg CO}_2/\text{m}^3$) based on unit rates and emission factors (OPC: 0.82, Fly Ash: 0.02, Silica Fume: 0.03, Sands: 0.005, SP: 0.75).
* **M4.4 Firestore Logging:** Button to write active mix parameters, batch weights, costs, and carbon metrics to Firebase Firestore database.

### Module 5: Context-Aware AI Engineering Co-Pilot
* **M5.1 System Context Injection:** Injects live React application state ($q_1, q_2, p, W/B, SP\%, \text{WPD}, R^2, \tau_0, \text{Flow}, N_{\max}, f_{ck}$, cost, carbon) into system prompts behind the scenes.
* **M5.2 Streaming UI:** Chat interface rendered at the bottom with quick-prompt preset chips ("Diagnose Extrudability", "Increase Buildable Layers", "Lower Cost & Carbon").

---

## 4. Design & UI Requirements
* **Theme:** Modern Light Theme with Tailwind CSS.
* **Color Palette:**
  * Primary Accent: Indigo (`#4f46e5`, Tailwind `indigo-600`)
  * Secondary / Highlight Accent: Amber/Gold (`#f59e0b`, Tailwind `amber-500`)
  * Neutral Background: Slate (`#f8fafc`, Tailwind `slate-50`)
  * Surface Cards: Pure White (`#ffffff`) with subtle slate borders (`border-slate-200`).
* **Layout:**
  * Collapsible Sidebar on Left for Sliders & Input Controls.
  * Top Metric Cards Row (R², WPD, Yield Stress, Max Layers, Cost/m³, Carbon/m³).
  * Main Tabbed Workspace (Tab 1: PSD Curves & Diagnostics, Tab 2: 3D Surface & ML, Tab 3: Batching & Cost Sheet, Tab 4: AI Co-Pilot Chat).

---

## 5. Master Raw Materials & Particle Size Matrix (`psd_data.json`)
```json
{
  "sieve_diameters_um": [0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 45.00, 90.00, 150.00, 300.00, 600.00, 1180.00, 2360.00],
  "materials": {
    "opc_53":      {"name": "OPC 53 Grade Cement", "density_kg_m3": 3150, "cost_inr_kg": 8.0,  "co2_factor": 0.820, "psd_passing": [0.0, 0.0, 0.0, 1.0, 6.0, 14.0, 29.0, 48.0, 70.0, 91.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]},
    "fly_ash":     {"name": "Class F Fly Ash",     "density_kg_m3": 2200, "cost_inr_kg": 3.5,  "co2_factor": 0.020, "psd_passing": [0.0, 0.0, 0.0, 2.5, 8.0, 20.0, 48.0, 75.0, 92.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]},
    "silica_fume": {"name": "Silica Fume 920D",    "density_kg_m3": 2220, "cost_inr_kg": 32.0, "co2_factor": 0.030, "psd_passing": [0.0, 22.0, 78.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]},
    "fine_sand":   {"name": "Fine Sand (0.15-0.6)", "density_kg_m3": 2650, "cost_inr_kg": 1.8,  "co2_factor": 0.005, "psd_passing": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0, 100.0, 100.0, 100.0]},
    "medium_sand": {"name": "Medium Sand (0.6-2.36)","density_kg_m3": 2650, "cost_inr_kg": 1.8,  "co2_factor": 0.005, "psd_passing": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.0, 100.0]}
  },
  "liquid_admixtures": {
    "water": {"density_kg_m3": 1000, "cost_inr_l": 0.10, "co2_factor": 0.000},
    "pce_sp": {"density_kg_m3": 1090, "cost_inr_l": 160.0, "co2_factor": 0.750},
    "vma":    {"density_kg_m3": 1050, "cost_inr_l": 220.0, "co2_factor": 0.500}
  }
}