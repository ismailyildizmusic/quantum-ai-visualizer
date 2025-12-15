"""
Quantum AI Visualizer - Streamlit Web Application
TÜBİTAK 2204-A Project

Notes:
- Uses natural units: ħ = 1, m = 1 (dimensionless)
- Wavefunction is computed by solving boundary conditions for a 1D rectangular barrier
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple, professional styling
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f6f7fb;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.card {
    background: white;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 16px;
    padding: 16px 16px 6px 16px;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
}
.small {
    color: rgba(15, 23, 42, 0.65);
    font-size: 0.92rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="card" style="padding: 18px 18px 12px 18px;">
  <div style="display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div>
      <div style="font-size: 2.0rem; font-weight: 800; color:#0f172a;">⚛️ Quantum AI Visualizer</div>
      <div class="small">1D Rectangular Barrier • Interactive Quantum Tunneling</div>
    </div>
    <div class="small" style="text-align:right;">
      Natural units: ħ = 1, m = 1<br/>
      Educational & scientific visualization
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# -------------------------
# Physics: Solve wavefunction by boundary conditions
# -------------------------
def solve_scattering_coeffs(E: float, V0: float, L: float):
    """
    Solves for r, A, B, t in:
      Region I (x<0):     ψ = e^{ikx} + r e^{-ikx}
      Region II (0<x<L):  ψ = A e^{iqx} + B e^{-iqx}   (q can be real or imaginary)
      Region III (x>L):   ψ = t e^{ikx}

    Natural units: ħ=1, m=1 => k = sqrt(2E)
    If E < V0: q = iκ, κ = sqrt(2(V0-E))
    If E > V0: q = k2 = sqrt(2(E-V0))
    """
    if E <= 0:
        E = 1e-6

    k = np.sqrt(2.0 * E)  # region I & III (V=0)

    if E < V0:
        kappa = np.sqrt(2.0 * (V0 - E))
        q = 1j * kappa  # so e^{iqx} = e^{-kappa x}, e^{-iqx}=e^{+kappa x} (still fine)
    else:
        k2 = np.sqrt(2.0 * (E - V0))
        q = k2

    # Unknown vector: [r, A, B, t]
    # Boundary at x=0:
    # 1 + r = A + B
    # ik(1 - r) = i q (A - B)   (because d/dx of A e^{iqx} + B e^{-iqx} at 0 is i q (A - B))
    #
    # Boundary at x=L:
    # A e^{iqL} + B e^{-iqL} = t e^{ikL}
    # i q (A e^{iqL} - B e^{-iqL}) = i k t e^{ikL}

    e_qL = np.exp(1j * q * L)
    e_mqL = np.exp(-1j * q * L)
    e_kL = np.exp(1j * k * L)

    M = np.zeros((4, 4), dtype=np.complex128)
    b = np.zeros((4,), dtype=np.complex128)

    # Eq1: 1 + r - A - B = 0
    M[0, 0] = 1.0
    M[0, 1] = -1.0
    M[0, 2] = -1.0
    M[0, 3] = 0.0
    b[0] = -1.0

    # Eq2: ik(1 - r) - i q (A - B) = 0  => (-ik) r + (-i q)A + (+i q)B = -ik
    M[1, 0] = -1j * k
    M[1, 1] = -1j * q
    M[1, 2] = +1j * q
    M[1, 3] = 0.0
    b[1] = -1j * k

    # Eq3: A e^{iqL} + B e^{-iqL} - t e^{ikL} = 0
    M[2, 0] = 0.0
    M[2, 1] = e_qL
    M[2, 2] = e_mqL
    M[2, 3] = -e_kL
    b[2] = 0.0

    # Eq4: i q (A e^{iqL} - B e^{-iqL}) - i k t e^{ikL} = 0
    M[3, 0] = 0.0
    M[3, 1] = 1j * q * e_qL
    M[3, 2] = -1j * q * e_mqL
    M[3, 3] = -1j * k * e_kL
    b[3] = 0.0

    r, A, B, t = np.linalg.solve(M, b)

    # For equal potentials on left/right, transmission coefficient T = |t|^2
    # (More general form includes k_right/k_left; here they are equal.)
    T = float(np.clip(np.abs(t) ** 2, 0.0, 1.0))

    return r, A, B, t, T, k, q


def psi_x(x: np.ndarray, E: float, V0: float, L: float, phase: float):
    """
    Returns wavefunction ψ(x) with a global time-like phase factor exp(-i*phase).
    This changes Re(ψ) but keeps |ψ|^2 invariant (physically consistent for stationary states).
    """
    r, A, B, t, T, k, q = solve_scattering_coeffs(E, V0, L)

    psi = np.zeros_like(x, dtype=np.complex128)

    # Region I
    mask1 = x < 0
    psi[mask1] = np.exp(1j * k * x[mask1]) + r * np.exp(-1j * k * x[mask1])

    # Region II
    mask2 = (x >= 0) & (x <= L)
    psi[mask2] = A * np.exp(1j * q * x[mask2]) + B * np.exp(-1j * q * x[mask2])

    # Region III
    mask3 = x > L
    psi[mask3] = t * np.exp(1j * k * x[mask3])

    psi *= np.exp(-1j * phase)
    return psi


def build_figure(E: float, V0: float, L: float, phase: float):
    x = np.linspace(-3.0, L + 3.0, 900)

    # Potential
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0

    # Wavefunction
    psi = psi_x(x, E, V0, L, phase)
    prob = np.abs(psi) ** 2
    repsi = np.real(psi)

    # Transmission
    _, _, _, _, T, _, _ = solve_scattering_coeffs(E, V0, L)

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=(
            "Potential & Energy (top) • Real part of ψ (overlay)",
            f"Probability Density |ψ|² (bottom) • Transmission T = {T:.4f}",
        ),
    )

    # Top: potential
    fig.add_trace(
        go.Scatter(
            x=x,
            y=V,
            name="V(x)",
            fill="tozeroy",
            line=dict(width=2),
            opacity=0.35,
        ),
        row=1,
        col=1,
    )

    # Top: energy line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[E] * len(x),
            name="E",
            line=dict(width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Overlay Re(psi) scaled and centered near E for readability
    # (purely visualization scaling; physics lives in prob below)
    scale = 0.28 * max(1.0, V0)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=E + scale * repsi / (np.max(np.abs(repsi)) + 1e-9),
            name="Re(ψ) (scaled)",
            line=dict(width=2),
            opacity=0.9,
        ),
        row=1,
        col=1,
    )

    # Bottom: probability density
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prob,
            name="|ψ|²",
            fill="tozeroy",
            line=dict(width=2),
            opacity=0.35,
        ),
        row=2,
        col=1,
    )

    # barrier boundaries
    for r in [1, 2]:
        fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="gray", row=r, col=1)
        fig.add_vline(x=L, line_dash="dot", line_width=1, line_color="gray", row=r, col=1)

    fig.update_xaxes(title_text="Position x", row=2, col=1)
    fig.update_yaxes(title_text="Energy / Potential", row=1, col=1)
    fig.update_yaxes(title_text="Probability density", row=2, col=1)

    fig.update_layout(
        height=760,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=28, r=28, t=70, b=28),
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig, T


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.caption("All values are dimensionless (ħ = 1, m = 1).")

    energy = st.slider("Particle Energy E", 0.10, 2.00, 0.80, 0.01)
    barrier_height = st.slider("Barrier Height V₀", 1.00, 3.00, 1.50, 0.01)
    barrier_width = st.slider("Barrier Width L", 0.50, 2.50, 1.00, 0.01)

    st.divider()

    # Manual phase control (no auto rerun → no flicker, no excessive motion)
    phase = st.slider("Phase (for Re(ψ) visualization)", 0.0, float(2 * np.pi), 0.0, 0.01)

    st.divider()

    preset = st.radio("Presets", ["Custom", "Strong Tunneling", "Weak Tunneling", "Classical", "Critical"], index=0)

    if preset == "Strong Tunneling":
        energy, barrier_height, barrier_width = 0.90, 1.00, 0.50
    elif preset == "Weak Tunneling":
        energy, barrier_height, barrier_width = 0.25, 2.50, 2.00
    elif preset == "Classical":
        energy, barrier_height, barrier_width = 1.80, 1.20, 1.00
    elif preset == "Critical":
        energy, barrier_height, barrier_width = 1.50, 1.50, 1.00


# -------------------------
# Main view
# -------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig, T = build_figure(energy, barrier_height, barrier_width, phase)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        f"""
<div class="card">
  <div style="font-size:1.05rem; font-weight:700; color:#0f172a;">Results</div>
  <div style="margin-top:10px; font-size:2.4rem; font-weight:900; color:#0f172a;">T = {T:.4f}</div>
  <div class="small" style="margin-top:4px;">Transmission probability</div>
  <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(15,23,42,0.10);">
    <div class="small">Regime</div>
    <div style="font-weight:700; margin-top:4px;">
      {"Classical / Above barrier (E ≥ V₀)" if energy >= barrier_height else "Quantum tunneling (E < V₀)"}
    </div>
  </div>
  <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(15,23,42,0.10);">
    <div class="small">Quick ratios</div>
    <div style="margin-top:6px;">E/V₀ = <b>{energy/barrier_height:.3f}</b></div>
    <div>Reflection ≈ <b>{(1-T)*100:.1f}%</b></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander("✅ Scientific notes (important)"):
        st.markdown(
            """
- This app computes ψ(x) by **solving boundary conditions** for a rectangular barrier (1D).
- We use **natural units**: ħ = 1, m = 1. (That’s why k = √(2E).)
- The **phase slider** changes Re(ψ) visually; **|ψ|² is invariant**, which is physically correct for stationary states.
- T is computed as **|t|²** (left/right potentials equal).
"""
        )

st.write("")
st.caption("Tip: If you want less visual complexity, keep phase = 0 and use presets. No auto-animation is running." )
