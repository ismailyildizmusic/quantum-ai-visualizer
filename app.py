"""
Quantum AI Visualizer - Streamlit Web Application
TÜBİTAK 2204-A Project
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- SAYFA AYARLARI (En üstte olmalı) ---
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GELİŞMİŞ CSS TASARIMI (Yanıp sönmeyi engeller) ---
st.markdown("""
<style>
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
        background-attachment: fixed;
    }
    
    /* Metrik Kutuları (Glassmorphism) */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Yazı Renkleri */
    h1, h2, h3, h4, p, li, span {
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Yan Menü Tasarımı */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(20px);
    }
    
    /* Butonlar */
    div.stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# --- BAŞLIK ALANI ---
st.markdown("""
<div style='text-align: center; padding: 3rem; background: rgba(0,0,0,0.3); border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(255,255,255,0.1);'>
    <h1 style='font-size: 3.5rem; text-shadow: 2px 2px 10px rgba(0,0,0,0.5);'>⚛️ Quantum AI Visualizer</h1>
    <p style='font-size: 1.4rem; opacity: 0.9;'>Yapay Zeka Destekli Kuantum Tünelleme Simülasyonu</p>
    <div style='margin-top: 15px;'>
        <span style='background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; font-size: 0.9rem;'>🏆 TÜBİTAK 2204-A Projesi</span>
        <span style='background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; margin-left: 10px;'>v2.0 Stable</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- OTURUM DURUMU ---
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False
if 'frame' not in st.session_state:
    st.session_state.frame = 0

# --- YAN MENÜ ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Schrodinger_cat.svg/1200px-Schrodinger_cat.svg.png", width=100)
    st.header("⚙️ Kontrol Paneli")
    
    st.subheader("📊 Fiziksel Parametreler")
    
    energy = st.slider("⚡ Parçacık Enerjisi (E)", 0.1, 3.0, 0.8, 0.01, format="%.2f eV")
    barrier_height = st.slider("📈 Bariyer Yüksekliği (V₀)", 1.0, 4.0, 1.5, 0.01, format="%.2f eV")
    barrier_width = st.slider("↔️ Bariyer Genişliği (L)", 0.5, 3.0, 1.0, 0.01, format="%.2f nm")
    
    st.markdown("---")
    st.subheader("🎬 Animasyon Kontrolü")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ Başlat / Durdur", use_container_width=True):
            st.session_state.animation_running = not st.session_state.animation_running
            
    with col_btn2:
        if st.button("🔄 Sıfırla", use_container_width=True):
            st.session_state.frame = 0
            st.session_state.animation_running = False
            
    st.markdown("---")
    st.info("💡 **İpucu:** Enerji bariyerden düşük olsa bile parçacığın geçme ihtimali vardır (Tünelleme).")

# --- HESAPLAMA MOTORU ---
def calculate_transmission(E, V0, L):
    """Schrödinger denklemi çözümleri"""
    if E >= V0:
        k1 = np.sqrt(2 * E)
        k2 = np.sqrt(2 * (E - V0))
        denom = (k1 + k2)**2 - (k1 - k2)**2 * np.sin(k2 * L)**2
        return (4 * k1 * k2 / denom) if denom != 0 else 1.0
    else:
        kappa = np.sqrt(2 * (V0 - E))
        if kappa * L > 50: return 0 # Taşmayı önle
        sinh_sq = np.sinh(kappa * L)**2
        denom = 4 * E * (V0 - E)
        return (1 / (1 + (V0**2 * sinh_sq) / denom)) if denom != 0 else 0.0

# --- GRAFİK OLUŞTURUCU ---
def create_frame(E, V0, L, frame):
    x = np.linspace(-3, L+3, 600)
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    
    T = calculate_transmission(E, V0, L)
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Potansiyel Enerji Profili", f"Olasılık Yoğunluğu (|ψ|²)")
    )
    
    # 1. Grafik: Potansiyel
    fig.add_trace(go.Scatter(
        x=x, y=V, name="Bariyer", fill='tozeroy',
        line=dict(color='#00d2ff', width=0), fillcolor='rgba(0, 210, 255, 0.2)'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x, y=[E]*len(x), name="Enerji Seviyesi",
        line=dict(color='#ff6b6b', width=3, dash='dash')
    ), row=1, col=1)
    
    # Animasyonlu Dalga
    phase = frame * 0.2
    wave_amp = []
    for xi in x:
        if xi < 0:
            val = E + 0.4 * np.cos(8 * xi - phase)
        elif 0 <= xi <= L:
            decay = np.exp(-np.sqrt(2 * max(V0 - E, 0.01)) * xi)
            val = E + 0.4 * decay * np.cos(phase) # Sönümlenme efekti
        else:
            val = E + 0.4 * np.sqrt(T) * np.cos(8 * (xi - L) - phase)
        wave_amp.append(val)
        
    fig.add_trace(go.Scatter(
        x=x, y=wave_amp, name="Dalga Fonksiyonu",
        line=dict(color='#ffffff', width=2), opacity=0.8
    ), row=1, col=1)
    
    # 2. Grafik: Olasılık
    kappa = np.sqrt(2 * max(V0 - E, 0.01))
    prob = []
    for xi in x:
        if xi < 0: p = 1.0 + 0.1 * np.sin(phase) # Gelen dalga titreşimi
        elif xi <= L: p = np.exp(-2 * kappa * xi)
        else: p = T
        prob.append(p)
        
    fig.add_trace(go.Scatter(
        x=x, y=prob, name="Olasılık", fill='tozeroy',
        line=dict(color='#fdbb2d', width=2), fillcolor='rgba(253, 187, 45, 0.3)'
    ), row=2, col=1)
    
    # Grafik Düzeni
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig, T

# --- ANA EKRAN DÜZENİ ---
col_main, col_stats = st.columns([3, 1])

with col_main:
    # Grafiği tutacak boş bir kutu oluşturuyoruz (Titreşimi önleyen sır bu!)
    chart_placeholder = st.empty()
    
    # Animasyon Mantığı
    if st.session_state.animation_running:
        st.session_state.frame += 1
        # Animasyonlu Enerji (Sinüs dalgası şeklinde hafif oynar)
        anim_E = energy + 0.1 * np.sin(st.session_state.frame * 0.1)
        anim_E = max(0.1, min(3.0, anim_E))
        
        fig, T = create_frame(anim_E, barrier_height, barrier_width, st.session_state.frame)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.05) # İşlemciyi rahatlat, animasyonu yumuşat
        st.rerun() # Sadece grafiği güncellemek için döngü
    else:
        # Animasyon durduğunda normal çizim
        fig, T = create_frame(energy, barrier_height, barrier_width, st.session_state.frame)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.markdown("### 📊 Anlık Veriler")
    
    T_final = calculate_transmission(energy, barrier_height, barrier_width)
    
    st.markdown(f"""
    <div style='background: rgba(0,0,0,0.4); padding: 15px; border-radius: 10px; border-left: 5px solid #00d2ff;'>
        <h4 style='margin:0; color: #aaa !important;'>Geçiş Olasılığı</h4>
        <h1 style='margin:0; font-size: 2.5rem; color: #00d2ff !important;'>%{T_final*100:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Enerji Oranı", f"{energy/barrier_height:.2f}")
    col_stat2.metric("Yansıma", f"%{(1-T_final)*100:.1f}")
    
    st.metric("De Broglie Dalgaboyu", f"{1.226/np.sqrt(energy):.3f} nm")
    
    if energy < barrier_height:
        st.warning("⚠️ Kuantum Tünelleme Aktif")
        st.markdown("*Parçacık bariyerden 'sızarak' geçiyor.*")
    else:
        st.success("✅ Klasik Geçiş Bölgesi")
        st.markdown("*Parçacık bariyerin üzerinden atlıyor.*")

# --- ALT BİLGİ VE FORMÜLLER (Eski kodunuzdaki detaylar geri geldi) ---
st.markdown("---")
with st.expander("📚 Teorik Arkaplan ve Formüller (Detaylı Bilgi)"):
    st.markdown("""
    ### Kuantum Tünelleme Nedir?
    Klasik fizikte, bir topun enerjisi bir tepenin yüksekliğinden azsa, top o tepeyi asla aşamaz. 
    Ancak kuantum mekaniğinde, parçacıklar dalga özelliği gösterdiği için bariyerin içinden "tünel açarak" geçebilirler.
    
    #### Matematiksel Model (Schrödinger Denklemi)
    Bu simülasyon, zamandan bağımsız Schrödinger denkleminin çözümüne dayanır:
    """)
    
    st.latex(r"-\frac{\hbar^2}{2m} \frac{d^2\psi}{dx^2} + V(x)\psi = E\psi")
    
    st.markdown("#### Geçiş Olasılığı Formülü (T)")
    st.latex(r"T = \left[ 1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)} \right]^{-1}")
    st.markdown("Burada $\kappa$ (kappa), dalga sönüm katsayısıdır:")
    st.latex(r"\kappa = \frac{\sqrt{2m(V_0-E)}}{\hbar}")

st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 50px;'>
    2025 © Dr. İsmail Yıldız - Adapazarı BİLSEM - TÜBİTAK 2204-A Projesi<br>
    Python & Streamlit & Plotly ile geliştirilmiştir.
</div>
""", unsafe_allow_html=True)
