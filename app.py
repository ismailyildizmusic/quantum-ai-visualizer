"""
Quantum AI Visualizer - Streamlit Web Application
TÜBİTAK 2204-A Project
"""

import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #667eea;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    """
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 20px; margin-bottom: 2rem;'>
    <h1 style='color: white; font-size: 3rem;'>⚛️ Quantum AI Visualizer</h1>
    <p style='color: white; font-size: 1.2rem;'>Interactive Quantum Tunneling Simulator</p>
    <p style='color: rgba(255,255,255,0.8);'>🏆 TÜBİTAK 2204-A Award Winner</p>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "animation" not in st.session_state:
    st.session_state.animation = False
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "fps" not in st.session_state:
    st.session_state.fps = 20  # default FPS


# Quantum calculations
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

    # Calculate transmission
    T = calculate_transmission(E, V0, L)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Potential Energy Profile", f"Probability Density (T = {T:.4f})"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Top plot: barrier
    fig.add_trace(
        go.Scatter(
            x=x,
            y=V,
            name="Potential Barrier",
            fill="tozeroy",
            fillcolor="rgba(102,126,234,0.3)",
            line=dict(color="#667eea", width=2),
        ),
        row=1,
        col=1,
    )

    # Top plot: energy line
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

    # Optional: wave animation overlay (purely illustrative)
    if show_wave:
        phase = frame * 0.35
        wave = np.zeros_like(x, dtype=float)

        for i, xi in enumerate(x):
            if xi < 0:
                wave[i] = E + 0.28 * np.cos(5.5 * xi + phase)
            elif 0 <= xi <= L:
                # decaying in barrier when E < V0, otherwise oscillatory
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
                line=dict(color="#4ecdc4", width=2),
                opacity=0.85,
            ),
            row=1,
            col=1,
        )

    # Bottom plot: probability density (simple illustrative model)
    prob = np.ones_like(x, dtype=float)
    if E < V0:
        kappa = np.sqrt(2 * max(V0 - E, 0.01))
        prob[x < 0] = 1.0
        mid = (x >= 0) & (x <= L)
        prob[mid] = np.exp(-2 * kappa * x[mid])
        prob[x > L] = T
    else:
        # if E >= V0, keep probability ~1 with a mild step after barrier
        prob[x > L] = min(1.0, 0.7 + 0.3 * T)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=prob,
            name="Probability |ψ|²",
            fill="tozeroy",
            fillcolor="rgba(168,85,247,0.3)",
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

    fig.update_layout(
        height=720,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=30),
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

    # FPS control (makes animation visibly smooth and prevents "freeze")
    st.session_state.fps = st.slider("🎞️ Animation FPS", 5, 40, st.session_state.fps, 1)

    show_wave = st.toggle("🌊 Show Wave Overlay", value=True)

    st.divider()

    col1b, col2b = st.columns(2)
    with col1b:
        if st.button(
            "▶️ Animate" if not st.session_state.animation else "⏸️ Pause",
            use_container_width=True,
        ):
            st.session_state.animation = not st.session_state.animation
    with col2b:
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


# Main content
left, right = st.columns([2, 1])

with left:
    # Visible frame indicator (helps user see it's running)
    frame_box = st.empty()

    # Compute animated energy (small oscillation)
    if st.session_state.animation:
        st.session_state.frame += 1
        animated_energy = energy + 0.20 * np.sin(st.session_state.frame * 0.12)
        animated_energy = float(np.clip(animated_energy, 0.1, 2.0))
        frame_box.caption(f"🎞️ Running… Frame: {st.session_state.frame} | FPS: {st.session_state.fps}")
    else:
        animated_energy = energy
        frame_box.caption(f"⏸️ Paused | Frame: {st.session_state.frame}")

    fig, T_anim = create_plot(
        animated_energy,
        barrier_height,
        barrier_width,
        frame=st.session_state.frame,
        show_wave=bool(show_wave and st.session_state.animation),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Controlled rerun for animation (this is the key fix)
    if st.session_state.animation:
        time.sleep(1.0 / max(st.session_state.fps, 1))
        st.rerun()

with right:
    st.markdown("### 📊 Results")

    # Use the "current" chosen sliders for reported stats (not the oscillating energy)
    T = calculate_transmission(energy, barrier_height, barrier_width)

    st.markdown(
        f"""
<div style='background: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;'>
    <h4 style='color: #667eea;'>Tunneling Probability</h4>
    <h1 style='color: #667eea; font-size: 3rem; margin: 10px 0;'>{T:.4f}</h1>
    <h3 style='color: #764ba2;'>{T*100:.2f}%</h3>
</div>
""",
        unsafe_allow_html=True,
    )

    if energy >= barrier_height:
        st.success("⚡ Classical Transmission (E ≥ V₀)")
    else:
        st.warning("🌊 Quantum Tunneling (E < V₀)")

    st.markdown("### 📈 Statistics")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("E/V₀ Ratio", f"{energy / barrier_height:.3f}")
        st.metric("Reflection", f"{(1 - T) * 100:.1f}%")
    with c2:
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
<div style='text-align: center; color: white; padding: 20px;'>
    <p>Created with ❤️ for TÜBİTAK 2204-A Science Fair</p>
    <p>⚛️ Quantum Physics + 🤖 Artificial Intelligence = 🚀 Future of Education</p>
</div>
""",
    unsafe_allow_html=True,
)
