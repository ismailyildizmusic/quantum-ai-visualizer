"""
Quantum AI Visualizer - Streamlit Web Application
TÜBİTAK 2204-A Project
"""

import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Strong anti-flicker CSS (locks background at html/body/container level) ---
st.markdown(
    """
<style>
/* Force background everywhere (prevents white flash on rerun) */
html, body, #root,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stVerticalBlock"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

/* Remove header bar fill */
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: rgba(0,0,0,0) !important;
}

/* Optional: reduce layout jump */
section.main > div {
    padding-top: 1rem;
}

/* Cards */
.q-card {
    background: rgba(255, 255, 255, 0.92);
    border: 2px solid rgba(102,126,234,0.55);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
}

/* Metrics */
div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.92);
    border: 2px solid rgba(102,126,234,0.55);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    """
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.10); border-radius: 20px; margin-bottom: 1.3rem;'>
    <h1 style='color: white; font-size: 3rem; margin-bottom: 0.3rem;'>⚛️ Quantum AI Visualizer</h1>
    <p style='color: white; font-size: 1.15rem; margin: 0;'>Interactive Quantum Tunneling Simulator</p>
    <p style='color: rgba(255,255,255,0.85); margin-top: 0.5rem;'>🏆 TÜBİTAK 2204-A Award Winner</p>
</div>
""",
    unsafe_allow_html=True,
)

# Session state
if "animation" not in st.session_state:
    st.session_state.animation = False
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "fps" not in st.session_state:
    st.session_state.fps = 20  # default FPS


def calculate_transmission(E, V0, L):
    """Calculate tunneling probability (0..1)"""
    if E >= V0:
        k1 = np.sqrt(2 * E)
        k2 = np.sqrt(2 * (E - V0))
        numerator = 4 * k1 * k2
        denominator = (k1 + k2) ** 2 - (k1 - k2) ** 2 * (np.sin(k2 * L) ** 2)
        T = numerator / denominator if denominator != 0 else 1.0
    else:
        kappa = np.sqrt(2 * (V0 - E))
        if kappa * L > 100:
            T = 16 * (E / V0) * (1 - E / V0) * np.exp(-2 * kappa * L)
        else:
            sinh_term = np.sinh(kappa * L) ** 2
            denom = 4 * E * (V0 - E)
            T = 1 / (1 + (V0**2 * sinh_term) / denom) if denom != 0 else 0.0
    return float(np.clip(T, 0.0, 1.0))


def create_plot(E, V0, L, frame=0, show_wave=False):
    """Create interactive Plotly visualization"""
    x = np.linspace(-3, L + 3, 600)

    # Potential barrier
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0

    # Transmission
    T = calculate_transmission(E, V0, L)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Potential Energy Profile", f"Probability Density (T = {T:.4f})"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=V,
            name="Potential Barrier",
            fill="tozeroy",
            fillcolor="rgba(102,126,234,0.28)",
            line=dict(color="#667eea", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=[E] * len(x),
            name=f"Energy (E={E:.2f})",
            line=dict(color="#ff6b6b", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Optional wave overlay (illustrative)
    if show_wave:
        phase = frame * 0.35
        wave = np.zeros_like(x, dtype=float)

        for i, xi in enumerate(x):
            if xi < 0:
                wave[i] = E + 0.28 * np.cos(5.5 * xi + phase)
            elif 0 <= xi <= L:
                if E < V0:
                    wave[i] = E + 0.28 * np.exp(-np.sqrt(2 * max(V0 - E, 0.01)) * xi)
                else:
                    wave[i] = E + 0.20 * np.cos(4.5 * xi + phase)
            else:
                wave[i] = E + 0.28 * np.sqrt(T) * np.cos(5.5 * (xi - L) + phase)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=wave,
                name="Wave Function (illustrative)",
                line=dict(color="#2fb6aa", width=2),
                opacity=0.9,
            ),
            row=1,
            col=1,
        )

    # Probability density
    prob = np.ones_like(x, dtype=float)
    if E < V0:
        kappa = np.sqrt(2 * max(V0 - E, 0.01))
        prob[x < 0] = 1.0
        mid = (x >= 0) & (x <= L)
        prob[mid] = np.exp(-2 * kappa * x[mid])
        prob[x > L] = T
    else:
        prob[x > L] = min(1.0, 0.7 + 0.3 * T)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=prob,
            name="Probability |ψ|²",
            fill="tozeroy",
            fillcolor="rgba(168,85,247,0.25)",
            line=dict(color="#a855f7", width=2),
        ),
        row=2,
        col=1,
    )

    # Barrier boundaries
    for r in [1, 2]:
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray", row=r, col=1)
        fig.add_vline(x=L, line_width=1, line_dash="dot", line_color="gray", row=r, col=1)

    fig.update_xaxes(title_text="Position (x)", row=2, col=1)
    fig.update_yaxes(title_text="Energy", row=1, col=1)
    fig.update_yaxes(title_text="|ψ|²", row=2, col=1)

    # IMPORTANT: prevent white flashing backgrounds in Plotly redraw
    fig.update_layout(
        height=720,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        margin=dict(l=28, r=28, t=60, b=28),
        font=dict(size=12),
    )

    return fig, T


# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")

    st.subheader("📊 Parameters")

    energy = st.slider(
        "⚡ Particle Energy (E)",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.01,
        format="%.2f",
    )

    barrier_height = st.slider(
        "📈 Barrier Height (V₀)",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.01,
        format="%.2f",
    )

    barrier_width = st.slider(
        "↔️ Barrier Width (L)",
        min_value=0.5,
        max_value=2.5,
        value=1.0,
        step=0.01,
        format="%.2f",
    )

    st.divider()

    st.session_state.fps = st.slider("🎞️ Animation FPS", 5, 40, st.session_state.fps, 1)
    show_wave = st.toggle("🌊 Show Wave Overlay", value=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "▶️ Animate" if not st.session_state.animation else "⏸️ Pause",
            use_container_width=True,
        ):
            st.session_state.animation = not st.session_state.animation
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.frame = 0
            st.session_state.animation = False

    st.divider()

    st.subheader("🎭 Presets")
    preset = st.radio(
        "Select scenario:",
        ["Custom", "Strong Tunneling", "Weak Tunneling", "Classical", "Critical Point"],
    )

    if preset == "Strong Tunneling":
        energy, barrier_height, barrier_width = 0.9, 1.0, 0.5
    elif preset == "Weak Tunneling":
        energy, barrier_height, barrier_width = 0.3, 2.5, 2.0
    elif preset == "Classical":
        energy, barrier_height, barrier_width = 1.8, 1.2, 1.0
    elif preset == "Critical Point":
        energy, barrier_height, barrier_width = 1.5, 1.5, 1.0


# Main layout
left, right = st.columns([2, 1])

with left:
    frame_box = st.empty()

    if st.session_state.animation:
        st.session_state.frame += 1
        animated_energy = energy + 0.20 * np.sin(st.session_state.frame * 0.12)
        animated_energy = float(np.clip(animated_energy, 0.1, 2.0))
        frame_box.caption(f"🎞️ Running… Frame: {st.session_state.frame} | FPS: {st.session_state.fps}")
    else:
        animated_energy = energy
        frame_box.caption(f"⏸️ Paused | Frame: {st.session_state.frame}")

    # Plot inside a card (reduces perceived flashing)
    st.markdown("<div class='q-card'>", unsafe_allow_html=True)
    fig, _ = create_plot(
        animated_energy,
        barrier_height,
        barrier_width,
        frame=st.session_state.frame,
        show_wave=bool(show_wave and st.session_state.animation),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Controlled rerun
    if st.session_state.animation:
        time.sleep(1.0 / max(st.session_state.fps, 1))
        st.rerun()

with right:
    st.markdown("### 📊 Results")

    T = calculate_transmission(energy, barrier_height, barrier_width)

    st.markdown(
        f"""
<div class='q-card' style='text-align:center;'>
    <h4 style='color: #667eea; margin: 0;'>Tunneling Probability</h4>
    <div style='font-size: 3rem; font-weight: 800; color: #667eea; margin: 0.2rem 0;'>{T:.4f}</div>
    <div style='font-size: 1.25rem; font-weight: 700; color: #764ba2;'>{T*100:.2f}%</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    if energy >= barrier_height:
        st.success("⚡ Classical Transmission (E ≥ V₀)")
    else:
        st.warning("🌊 Quantum Tunneling (E < V₀)")

    st.markdown("### 📈 Statistics")

    a, b = st.columns(2)
    with a:
        st.metric("E/V₀ Ratio", f"{energy / barrier_height:.3f}")
        st.metric("Reflection", f"{(1 - T) * 100:.1f}%")
    with b:
        if energy < barrier_height:
            penetration = 1 / np.sqrt(2 * (barrier_height - energy))
            st.metric("Penetration", f"{penetration:.3f}")
        else:
            st.metric("Penetration", "∞")
        st.metric("Barrier Factor", f"{barrier_height * barrier_width:.2f}")

    with st.expander("ℹ️ About Quantum Tunneling"):
        st.markdown(
            """
**Quantum tunneling** is a quantum mechanical phenomenon where particles pass through energy barriers that they classically shouldn't be able to cross.

**Applications:**
- ⚡ Transistors
- ☀️ Nuclear fusion in stars
- 🔬 Scanning tunneling microscope
- 💻 Quantum computers
"""
        )

    with st.expander("📐 Mathematical Formulas"):
        if energy < barrier_height:
            st.latex(r"T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)}}")
            st.latex(r"\kappa = \sqrt{2(V_0-E)}")
        else:
            st.latex(r"T = \frac{4k_1k_2}{(k_1+k_2)^2 - (k_1-k_2)^2\sin^2(k_2L)}")

# Footer
st.divider()
st.markdown(
    """
<div style='text-align: center; color: white; padding: 20px; background: rgba(255,255,255,0.08); border-radius: 14px;'>
    <p style='margin: 0;'>Created with ❤️ for TÜBİTAK 2204-A Science Fair</p>
    <p style='margin: 0.35rem 0 0 0;'>⚛️ Quantum Physics + 🤖 Artificial Intelligence = 🚀 Future of Education</p>
</div>
""",
    unsafe_allow_html=True,
)
