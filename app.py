"""
Quantum AI Visualizer - Streamlit Web Application
TÜBİTAK 2204-A Project
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS DÜZELTMESİ YAPILDI (Tasarım Aynen Korundu) ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #667eea;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3, p, span {
        color: white !important;
    }
    /* Metrik etiketlerini okunur yap */
    div[data-testid="stMetricLabel"] p {
        color: #4a5568 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #764ba2 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 20px; margin-bottom: 2rem;'>
    <h1 style='color: white; font-size: 3rem;'>⚛️ Quantum AI Visualizer</h1>
    <p style='color: white; font-size: 1.2rem;'>Interactive Quantum Tunneling Simulator</p>
    <p style='color: rgba(255,255,255,0.8);'>🏆 TÜBİTAK 2204-A Award Winner</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'animation' not in st.session_state:
    st.session_state.animation = False
if 'frame' not in st.session_state:
    st.session_state.frame = 0

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    st.subheader("📊 Parameters")
    
    # Energy slider
    energy = st.slider(
        "⚡ Particle Energy (E)",
        min_value=0.1, max_value=2.0, value=0.8, step=0.01, format="%.2f"
    )
    
    # Barrier height slider
    barrier_height = st.slider(
        "📈 Barrier Height (V₀)",
        min_value=1.0, max_value=3.0, value=1.5, step=0.01, format="%.2f"
    )
    
    # Barrier width slider
    barrier_width = st.slider(
        "↔️ Barrier Width (L)",
        min_value=0.5, max_value=2.5, value=1.0, step=0.01, format="%.2f"
    )
    
    st.divider()
    
    # Animation control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Animate" if not st.session_state.animation else "⏸️ Pause", use_container_width=True):
            st.session_state.animation = not st.session_state.animation
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.frame = 0
            st.session_state.animation = False
    
    st.divider()
    
    # Presets
    st.subheader("🎭 Presets")
    preset = st.radio(
        "Select scenario:",
        ["Custom", "Strong Tunneling", "Weak Tunneling", "Classical", "Critical Point"]
    )
    
    if preset == "Strong Tunneling":
        energy, barrier_height, barrier_width = 0.9, 1.0, 0.5
    elif preset == "Weak Tunneling":
        energy, barrier_height, barrier_width = 0.3, 2.5, 2.0
    elif preset == "Classical":
        energy, barrier_height, barrier_width = 1.8, 1.2, 1.0
    elif preset == "Critical Point":
        energy, barrier_height, barrier_width = 1.5, 1.5, 1.0

# Quantum calculations
def calculate_transmission(E, V0, L):
    """Calculate tunneling probability"""
    if E >= V0:
        k1 = np.sqrt(2 * E)
        k2 = np.sqrt(2 * (E - V0))
        numerator = 4 * k1 * k2
        denominator = (k1 + k2)**2 - (k1 - k2)**2 * np.sin(k2 * L)**2
        T = numerator / denominator if denominator != 0 else 1
    else:
        kappa = np.sqrt(2 * (V0 - E))
        if kappa * L > 100:
            T = 16 * (E / V0) * (1 - E / V0) * np.exp(-2 * kappa * L)
        else:
            sinh_term = np.sinh(kappa * L)**2
            denominator = 4 * E * (V0 - E)
            T = 1 / (1 + (V0**2 * sinh_term) / denominator) if denominator != 0 else 0
    return min(max(T, 0), 1)

# Create visualization
def create_plot(E, V0, L, frame=0):
    """Create interactive Plotly visualization"""
    x = np.linspace(-3, L+3, 500)
    
    # Potential barrier
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    
    # Calculate transmission
    T = calculate_transmission(E, V0, L)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Potential Energy Profile", f"Probability Density (T = {T:.4f})"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Top plot - Potential and Energy
    fig.add_trace(
        go.Scatter(x=x, y=V, name="Potential Barrier", 
                   fill='tozeroy', fillcolor='rgba(102,126,234,0.3)',
                   line=dict(color='#667eea', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=x, y=[E]*len(x), name=f"Energy (E={E:.2f})",
                   line=dict(color='#ff6b6b', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Wave function animation
    if st.session_state.animation and frame > 0:
        phase = frame * 0.1
        wave = []
        for xi in x:
            if xi < 0:
                wave.append(E + 0.3 * np.cos(5 * xi + phase))
            elif 0 <= xi <= L:
                wave.append(E + 0.3 * np.exp(-np.sqrt(2 * max(V0 - E, 0.01)) * xi))
            else:
                wave.append(E + 0.3 * np.sqrt(T) * np.cos(5 * (xi - L) + phase))
        
        fig.add_trace(
            go.Scatter(x=x, y=wave, name="Wave Function",
                       line=dict(color='#4ecdc4', width=2),
                       opacity=0.7),
            row=1, col=1
        )
    
    # Bottom plot - Probability density
    prob = np.ones_like(x)
    if E < V0:
        kappa = np.sqrt(2 * max(V0 - E, 0.01))
        prob[x < 0] = 1.0
        prob[(x >= 0) & (x <= L)] = np.exp(-2 * kappa * x[(x >= 0) & (x <= L)])
        prob[x > L] = T
    
    fig.add_trace(
        go.Scatter(x=x, y=prob, name="Probability |ψ|²",
                   fill='tozeroy', fillcolor='rgba(168,85,247,0.3)',
                   line=dict(color='#a855f7', width=2)),
        row=2, col=1
    )
    
    # Add barrier boundaries
    for row in [1, 2]:
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray", row=row, col=1)
        fig.add_vline(x=L, line_width=1, line_dash="dot", line_color="gray", row=row, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Position (x)", row=2, col=1)
    fig.update_yaxes(title_text="Energy", row=1, col=1)
    fig.update_yaxes(title_text="|ψ|²", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template="plotly_white",
        font=dict(size=12)
    )
    
    return fig, T

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Handle animation
    if st.session_state.animation:
        st.session_state.frame += 1
        animated_energy = energy + 0.2 * np.sin(st.session_state.frame * 0.1)
        animated_energy = max(0.1, min(2.0, animated_energy))
    else:
        animated_energy = energy
    
    # Create and display plot
    fig, T = create_plot(
        animated_energy if st.session_state.animation else energy,
        barrier_height,
        barrier_width,
        st.session_state.frame
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh for animation
    if st.session_state.animation:
        st.rerun()

with col2:
    st.markdown("### 📊 Results")
    
    # Calculate transmission
    T = calculate_transmission(energy, barrier_height, barrier_width)
    
    # Display probability
    st.markdown(f"""
    <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;'>
        <h4 style='color: #667eea; margin:0;'>Tunneling Probability</h4>
        <h1 style='color: #667eea; font-size: 3rem; margin: 10px 0;'>{T:.4f}</h1>
        <h3 style='color: #764ba2; margin:0;'>{T*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicator
    if energy >= barrier_height:
        st.success("⚡ Classical Transmission (E ≥ V₀)")
    else:
        st.warning("🌊 Quantum Tunneling (E < V₀)")
    
    # Statistics
    st.markdown("### 📈 Statistics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("E/V₀ Ratio", f"{energy/barrier_height:.3f}")
        st.metric("Reflection", f"{(1-T)*100:.1f}%")
    
    with col_b:
        if energy < barrier_height:
            penetration = 1/np.sqrt(2 * (barrier_height - energy))
            st.metric("Penetration", f"{penetration:.3f}")
        else:
            st.metric("Penetration", "∞")
        st.metric("Barrier Factor", f"{barrier_height * barrier_width:.2f}")
    
    # Info box
    with st.expander("ℹ️ About Quantum Tunneling"):
        st.markdown("""
        **Quantum tunneling** is a quantum mechanical phenomenon where 
        particles pass through energy barriers that they classically 
        shouldn't be able to cross.
        
        **Applications:**
        - ⚡ Transistors
        - ☀️ Nuclear fusion in stars
        - 🔬 Scanning tunneling microscope
        - 💻 Quantum computers
        """)
    
    # Formulas
    with st.expander("📐 Mathematical Formulas"):
        if energy < barrier_height:
            st.latex(r"T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)}}")
            st.latex(r"\kappa = \sqrt{2(V_0-E)}")
        else:
            st.latex(r"T = \frac{4k_1k_2}{(k_1+k_2)^2 - (k_1-k_2)^2\sin^2(k_2L)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>Created with ❤️ for TÜBİTAK 2204-A Science Fair</p>
    <p>⚛️ Quantum Physics + 🤖 Artificial Intelligence = 🚀 Future of Education</p>
</div>
""", unsafe_allow_html=True)
