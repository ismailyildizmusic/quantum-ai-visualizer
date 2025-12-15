"""
Quantum AI Visualizer - Streamlit Web Application
TÜBİTAK 2204-A Project
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time  # <--- YENİ EKLENDİ (Animasyon hızı için şart)

# Page config
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TASARIM ---
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
    
    energy = st.slider("⚡ Particle Energy (E)", 0.1, 2.0, 0.8, 0.01, format="%.2f")
    barrier_height = st.slider("📈 Barrier Height (V₀)", 1.0, 3.0, 1.5, 0.01, format="%.2f")
    barrier_width = st.slider("↔️ Barrier Width (L)", 0.5, 2.5, 1.0, 0.01, format="%.2f")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        # Buton mantığı
        if st.button("▶️ Animate" if not st.session_state.animation else "⏸️ Pause", use_container_width=True):
            st.session_state.animation = not st.session_state.animation
            st.rerun() # Tıklayınca hemen tepki ver
            
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.frame = 0
            st.session_state.animation = False
            st.rerun()
    
    st.divider()
    
    st.subheader("🎭 Presets")
    preset = st.radio("Select scenario:", ["Custom", "Strong Tunneling", "Weak Tunneling", "Classical"])
    
    if preset == "Strong Tunneling": energy, barrier_height, barrier_width = 0.9, 1.0, 0.5
    elif preset == "Weak Tunneling": energy, barrier_height, barrier_width = 0.3, 2.5, 2.0
    elif preset == "Classical": energy, barrier_height, barrier_width = 1.8, 1.2, 1.0

# Calculation Functions
def calculate_transmission(E, V0, L):
    if E >= V0:
        k1, k2 = np.sqrt(2*E), np.sqrt(2*(E-V0))
        denom = (k1+k2)**2 - (k1-k2)**2 * np.sin(k2*L)**2
        return (4*k1*k2 / denom) if denom != 0 else 1
    else:
        kappa = np.sqrt(2*(V0-E))
        sinh_sq = np.sinh(kappa*L)**2
        denom = 4*E*(V0-E)
        return (1 / (1 + (V0**2 * sinh_sq)/denom)) if denom != 0 else 0
    return 0

# Plot Function
def create_plot(E, V0, L, frame=0):
    x = np.linspace(-3, L+3, 500)
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    T = calculate_transmission(E, V0, L)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Potential Energy Profile", f"Probability Density (T = {T:.4f})"), vertical_spacing=0.12)
    
    # 1. Grafik
    fig.add_trace(go.Scatter(x=x, y=V, name="Potential", fill='tozeroy', fillcolor='rgba(102,126,234,0.3)', line=dict(color='#667eea', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[E]*len(x), name="Energy", line=dict(color='#ff6b6b', width=2, dash='dash')), row=1, col=1)
    
    # Animasyon Dalgaları
    if st.session_state.animation:
        phase = frame * 0.2  # Hız çarpanı
        wave = []
        for xi in x:
            if xi < 0: wave.append(E + 0.3 * np.cos(5*xi + phase))
            elif 0 <= xi <= L: wave.append(E + 0.3 * np.exp(-np.sqrt(2*max(V0-E, 0.01))*xi))
            else: wave.append(E + 0.3 * np.sqrt(T) * np.cos(5*(xi-L) + phase))
        fig.add_trace(go.Scatter(x=x, y=wave, name="Wave", line=dict(color='#4ecdc4', width=2), opacity=0.7), row=1, col=1)
    
    # 2. Grafik
    prob = np.ones_like(x)
    if E < V0:
        kappa = np.sqrt(2 * max(V0 - E, 0.01))
        prob = [1.0 if xi < 0 else (np.exp(-2*kappa*xi) if xi <= L else T) for xi in x]
    fig.add_trace(go.Scatter(x=x, y=prob, name="|ψ|²", fill='tozeroy', fillcolor='rgba(168,85,247,0.3)', line=dict(color='#a855f7', width=2)), row=2, col=1)
    
    for row in [1, 2]:
        fig.add_vline(x=0, line_dash="dot", line_color="gray", row=row, col=1)
        fig.add_vline(x=L, line_dash="dot", line_color="gray", row=row, col=1)
        
    fig.update_layout(height=700, showlegend=True, template="plotly_white")
    return fig, T

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Animasyon Hesaplaması
    if st.session_state.animation:
        st.session_state.frame += 1
        animated_E = energy + 0.2 * np.sin(st.session_state.frame * 0.1)
        animated_E = max(0.1, min(2.0, animated_E))
    else:
        animated_E = energy
        
    fig, T = create_plot(animated_E, barrier_height, barrier_width, st.session_state.frame)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- İŞTE SİHİRLİ DOKUNUŞ ---
    if st.session_state.animation:
        time.sleep(0.05)  # Saniyede 20 kare hızına sabitle
        st.rerun()       # Ekranı yenile

with col2:
    st.markdown("### 📊 Results")
    T_final = calculate_transmission(energy, barrier_height, barrier_width)
    
    st.markdown(f"""
    <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;'>
        <h4 style='color: #667eea; margin:0;'>Tunneling Probability</h4>
        <h1 style='color: #667eea; font-size: 3rem; margin: 10px 0;'>{T_final:.4f}</h1>
        <h3 style='color: #764ba2; margin:0;'>{T_final*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if energy >= barrier_height: st.success("⚡ Classical Transmission")
    else: st.warning("🌊 Quantum Tunneling")
    
    st.metric("E/V₀ Ratio", f"{energy/barrier_height:.3f}")
    st.metric("Reflection", f"{(1-T_final)*100:.1f}%")

st.divider()
st.markdown("<div style='text-align: center; color: white;'>Created for TÜBİTAK 2204-A</div>", unsafe_allow_html=True)
