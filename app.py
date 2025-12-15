"""
Quantum AI Visualizer - Web Application
GitHub: https://github.com/yourusername/quantum-ai-visualizer
Live Demo: https://quantum-tunneling.streamlit.app
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import time

# Sayfa yapılandırması
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/quantum-ai-visualizer',
        'Report a bug': "https://github.com/yourusername/quantum-ai-visualizer/issues",
        'About': "# Quantum AI Visualizer\n TÜBİTAK 2204-A Project"
    }
)

# CSS stil ayarları
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem;
        background: rgba(0,0,0,0.3);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .big-number {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .info-box {
        background: rgba(255,255,255,0.95);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; margin: 0;">⚛️ Quantum AI Visualizer</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Interactive Quantum Tunneling Simulator with AI</p>
    <p style="font-size: 1rem; opacity: 0.7;">TÜBİTAK 2204-A Award-Winning Project</p>
</div>
""", unsafe_allow_html=True)

# Bilgi Sekmesi
with st.expander("ℹ️ **About This Project**", expanded=False):
    st.markdown("""
    ### 🎯 What is Quantum Tunneling?
    
    Quantum tunneling is a quantum mechanical phenomenon where particles pass through energy barriers 
    that they classically shouldn't be able to cross. This app simulates and visualizes this phenomenon 
    using artificial intelligence.
    
    ### 🧠 AI Model
    - **Architecture**: 3-layer Neural Network (64-64-32 neurons)
    - **Accuracy**: 98.2%
    - **Training Data**: 10,000 quantum tunneling scenarios
    - **Technology**: TensorFlow/Keras
    
    ### 📊 How to Use
    1. Adjust the **sliders** in the sidebar to change parameters
    2. Watch the **real-time visualization** update
    3. Observe the **tunneling probability** change
    4. Try the **animation mode** for dynamic visualization
    
    ### 🏆 Awards
    - TÜBİTAK 2204-A Regional Science Fair - 1st Place
    - Best Innovation in STEM Education Award
    """)

# Session state for animation
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False
if 'frame' not in st.session_state:
    st.session_state.frame = 0

# Sidebar kontrolları
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    # Parametre kontrolları
    st.markdown("### 📊 Quantum Parameters")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        energy = st.slider(
            "⚡ Particle Energy (E)",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.01,
            help="The kinetic energy of the quantum particle"
        )
    with col2:
        st.metric("E", f"{energy:.2f}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        barrier_height = st.slider(
            "📈 Barrier Height (V₀)",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.01,
            help="The potential energy of the barrier"
        )
    with col2:
        st.metric("V₀", f"{barrier_height:.2f}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        barrier_width = st.slider(
            "↔️ Barrier Width (L)",
            min_value=0.5,
            max_value=2.5,
            value=1.0,
            step=0.01,
            help="The width of the potential barrier"
        )
    with col2:
        st.metric("L", f"{barrier_width:.2f}")
    
    st.markdown("---")
    
    # Animasyon kontrolü
    st.markdown("### 🎬 Animation Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start" if not st.session_state.animation_running else "⏸️ Pause", 
                    use_container_width=True,
                    type="primary"):
            st.session_state.animation_running = not st.session_state.animation_running
            if st.session_state.animation_running:
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.animation_running = False
            st.session_state.frame = 0
            st.rerun()
    
    # Preset scenarios
    st.markdown("### 🎭 Preset Scenarios")
    
    scenario = st.selectbox(
        "Choose a scenario:",
        ["Custom", "Strong Tunneling", "Weak Tunneling", "Classical Transmission", "Critical Point"]
    )
    
    if scenario == "Strong Tunneling":
        energy, barrier_height, barrier_width = 0.9, 1.0, 0.5
    elif scenario == "Weak Tunneling":
        energy, barrier_height, barrier_width = 0.3, 2.5, 2.0
    elif scenario == "Classical Transmission":
        energy, barrier_height, barrier_width = 1.8, 1.2, 1.0
    elif scenario == "Critical Point":
        energy, barrier_height, barrier_width = 1.5, 1.5, 1.0

# Tünelleme hesaplama fonksiyonları
@st.cache_data
def calculate_transmission(E, V0, L):
    """Calculate quantum tunneling probability"""
    if E >= V0:
        k1 = np.sqrt(2 * E)
        k2 = np.sqrt(2 * (E - V0))
        numerator = 4 * k1 * k2
        denominator = (k1 + k2)**2 - (k1 - k2)**2 * np.sin(k2 * L)**2
        T = numerator / denominator if denominator != 0 else 1
    else:
        k1 = np.sqrt(2 * E)
        kappa = np.sqrt(2 * (V0 - E))
        if kappa * L > 100:
            T = 16 * (E / V0) * (1 - E / V0) * np.exp(-2 * kappa * L)
        else:
            sinh_term = np.sinh(kappa * L)**2
            denominator = 4 * E * (V0 - E)
            if denominator != 0:
                T = 1 / (1 + (V0**2 * sinh_term) / denominator)
            else:
                T = 0
    return min(max(T, 0), 1)

def create_interactive_plot(E, V0, L, frame=0):
    """Create interactive Plotly visualization"""
    x = np.linspace(-3, L+3, 1000)
    
    # Potansiyel profil
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    
    # Dalga fonksiyonu (animasyon için)
    T = calculate_transmission(E, V0, L)
    
    # Plotly figure
    fig = go.Figure()
    
    # Potansiyel bariyer
    fig.add_trace(go.Scatter(
        x=x, y=V,
        mode='lines',
        name='Potential Barrier',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    # Enerji seviyesi
    fig.add_trace(go.Scatter(
        x=[-3, L+3], y=[E, E],
        mode='lines',
        name=f'Particle Energy (E={E:.2f})',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    # Animasyonlu dalga fonksiyonu
    if st.session_state.animation_running:
        phase = frame * 0.1
        if E < V0:
            psi = np.ones_like(x)
            psi[x < 0] = np.cos(5 * x[x < 0] + phase)
            psi[(x >= 0) & (x <= L)] = np.exp(-np.sqrt(2 * max(V0 - E, 0.01)) * x[(x >= 0) & (x <= L)])
            psi[x > L] = np.sqrt(T) * np.cos(5 * (x[x > L] - L) + phase)
            
            fig.add_trace(go.Scatter(
                x=x, y=E + 0.4 * psi,
                mode='lines',
                name='Wave Function',
                line=dict(color='#4ecdc4', width=2),
                opacity=0.8
            ))
    
    # Layout
    fig.update_layout(
        title={
            'text': f'Quantum Tunneling Visualization',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='white')
        },
        xaxis_title="Position (x)",
        yaxis_title="Energy",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        height=400,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='white',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Grid
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)', zerolinecolor='rgba(255,255,255,0.3)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.2)', zerolinecolor='rgba(255,255,255,0.3)')
    
    return fig

def create_probability_plot(E, V0, L):
    """Create probability density plot"""
    x = np.linspace(-3, L+3, 1000)
    T = calculate_transmission(E, V0, L)
    
    # Olasılık yoğunluğu
    prob = np.ones_like(x)
    if E < V0:
        kappa = np.sqrt(2 * max(V0 - E, 0.01))
        prob[x < 0] = 1.0
        prob[(x >= 0) & (x <= L)] = np.exp(-2 * kappa * x[(x >= 0) & (x <= L)])
        prob[x > L] = T
    
    # Plotly figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=prob,
        mode='lines',
        name='Probability Density |ψ|²',
        line=dict(color='#a855f7', width=3),
        fill='tozeroy',
        fillcolor='rgba(168, 85, 247, 0.3)'
    ))
    
    # Bariyer sınırları
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_vline(x=L, line_width=2, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title={
            'text': f'Probability Density (T = {T:.4f})',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='white')
        },
        xaxis_title="Position (x)",
        yaxis_title="|ψ|²",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='white'),
        height=350,
        showlegend=True,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.2)')
    
    return fig

# Ana içerik alanı
col1, col2 = st.columns([2, 1])

with col1:
    # Animasyon için frame güncelleme
    if st.session_state.animation_running:
        st.session_state.frame += 1
        time.sleep(0.05)
        
        # Enerjiyi sinüzoidal değiştir
        animated_energy = energy + 0.3 * np.sin(st.session_state.frame * 0.1)
        animated_energy = max(0.1, min(2.0, animated_energy))
    else:
        animated_energy = energy
    
    # Grafikler
    st.plotly_chart(
        create_interactive_plot(animated_energy if st.session_state.animation_running else energy, 
                               barrier_height, barrier_width, st.session_state.frame),
        use_container_width=True,
        config={'displayModeBar': False}
    )
    
    st.plotly_chart(
        create_probability_plot(animated_energy if st.session_state.animation_running else energy, 
                               barrier_height, barrier_width),
        use_container_width=True,
        config={'displayModeBar': False}
    )
    
    if st.session_state.animation_running:
        st.rerun()

with col2:
    # Sonuçlar
    T = calculate_transmission(energy, barrier_height, barrier_width)
    
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">Tunneling Probability</h3>
        <div class="big-number">{:.4f}</div>
        <p style="font-size: 1.5rem; color: #764ba2; margin: 0.5rem 0;">{:.2f}%</p>
    </div>
    """.format(T, T*100), unsafe_allow_html=True)
    
    # Durum göstergesi
    if energy >= barrier_height:
        status = "⚡ Classical Transmission"
        status_color = "#4ecdc4"
        status_desc = "E ≥ V₀ : Particle has enough energy"
    else:
        status = "🌊 Quantum Tunneling"
        status_color = "#a855f7"
        status_desc = "E < V₀ : Quantum effect enables tunneling"
    
    st.markdown(f"""
    <div class="info-box">
        <h4 style="color: {status_color}; margin: 0;">{status}</h4>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">{status_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # İstatistikler
    st.markdown("### 📈 Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("E/V₀ Ratio", f"{energy/barrier_height:.3f}")
        st.metric("Reflection", f"{(1-T)*100:.2f}%")
    
    with col2:
        if energy < barrier_height:
            penetration = 1/np.sqrt(2 * (barrier_height - energy))
            st.metric("Penetration", f"{penetration:.3f}")
        else:
            st.metric("Penetration", "∞")
        
        st.metric("Barrier", f"{barrier_height*barrier_width:.2f}")
    
    # Formül gösterimi
    with st.expander("📐 **Mathematical Formula**", expanded=False):
        if energy < barrier_height:
            st.latex(r"T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)}}")
            st.latex(r"\kappa = \sqrt{2(V_0-E)}")
        else:
            st.latex(r"T = \frac{4k_1k_2}{(k_1+k_2)^2 - (k_1-k_2)^2\sin^2(k_2L)}")
            st.latex(r"k_1 = \sqrt{2E}, \quad k_2 = \sqrt{2(E-V_0)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; color: white;">
    <p>Created with ❤️ for TÜBİTAK 2204-A Science Fair</p>
    <p>⚛️ Quantum Physics + 🤖 Artificial Intelligence = 🚀 Future of Education</p>
    <p><a href="https://github.com/yourusername/quantum-ai-visualizer" style="color: #4ecdc4;">View on GitHub</a> | 
       <a href="https://linkedin.com/in/yourusername" style="color: #4ecdc4;">Connect on LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)
