import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TASARIM KODLARI (HATAYI ÇÖZEN KISIM BURASI) ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #667eea;
        padding: 10px;
        border-radius: 10px;
    }
    h1, h2, h3, p {
        color: white !important;
    }
    .stMarkdown p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- BAŞLIK ALANI ---
st.markdown("""
<div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 20px; margin-bottom: 2rem;'>
    <h1 style='color: white; font-size: 3rem;'>⚛️ Quantum AI Visualizer</h1>
    <p style='color: white; font-size: 1.2rem;'>İnteraktif Kuantum Tünelleme Simülatörü</p>
    <p style='color: rgba(255,255,255,0.8);'>🏆 TÜBİTAK 2204-A Projesi</p>
</div>
""", unsafe_allow_html=True)

# --- OTURUM DURUMU ---
if 'animation' not in st.session_state:
    st.session_state.animation = False
if 'frame' not in st.session_state:
    st.session_state.frame = 0

# --- YAN MENÜ ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    st.subheader("📊 Parametreler")
    
    energy = st.slider("⚡ Parçacık Enerjisi (E)", 0.1, 2.0, 0.8, 0.01)
    barrier_height = st.slider("📈 Bariyer Yüksekliği (V₀)", 1.0, 3.0, 1.5, 0.01)
    barrier_width = st.slider("↔️ Bariyer Genişliği (L)", 0.5, 2.5, 1.0, 0.01)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Oynat/Duraklat", use_container_width=True):
            st.session_state.animation = not st.session_state.animation
    with col2:
        if st.button("🔄 Sıfırla", use_container_width=True):
            st.session_state.frame = 0
            st.session_state.animation = False

# --- HESAPLAMALAR VE GRAFİK ---
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

def create_plot(E, V0, L, frame):
    x = np.linspace(-3, L+3, 500)
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    T = calculate_transmission(E, V0, L)
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15,
                        subplot_titles=("Potansiyel Enerji", "Olasılık Yoğunluğu"))

    # Potansiyel Grafiği
    fig.add_trace(go.Scatter(x=x, y=V, fill='tozeroy', name="Bariyer", line_color='#667eea'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[E]*len(x), name="Enerji", line=dict(color='#ff6b6b', dash='dash')), row=1, col=1)
    
    # Animasyon Dalgaları
    if st.session_state.animation:
        phase = frame * 0.1
        wave = [E + 0.3 * np.cos(5*xi + phase) if xi < 0 else 
                (E + 0.3 * np.exp(-np.sqrt(2*max(V0-E, 0.01))*xi) if xi <= L else 
                 E + 0.3 * np.sqrt(T) * np.cos(5*(xi-L) + phase)) for xi in x]
        fig.add_trace(go.Scatter(x=x, y=wave, name="Dalga", line_color='#4ecdc4', opacity=0.7), row=1, col=1)

    # Olasılık Grafiği
    kappa = np.sqrt(2 * max(V0 - E, 0.01))
    prob = [1.0 if xi < 0 else (np.exp(-2 * kappa * xi) if xi <= L else T) for xi in x]
    fig.add_trace(go.Scatter(x=x, y=prob, fill='tozeroy', name="|ψ|²", line_color='#a855f7'), row=2, col=1)

    fig.update_layout(height=600, template="plotly_white", margin=dict(t=30,b=30))
    return fig, T

# --- ANA EKRAN ---
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.animation:
        st.session_state.frame += 1
        st.rerun()
    
    fig, T = create_plot(energy, barrier_height, barrier_width, st.session_state.frame)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📊 Sonuçlar")
    st.metric("Tünelleme Olasılığı", f"%{T*100:.2f}")
    if energy < barrier_height:
        st.warning("🌊 Kuantum Tünelleme")
    else:
        st.success("⚡ Klasik Geçiş")
