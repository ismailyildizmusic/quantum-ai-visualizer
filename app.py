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

# --- CSS TASARIM KODLARI ---
st.markdown("""
<style>
    /* Ana arkaplan rengi */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    /* Metrik kutuları */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid #667eea;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Başlık rengi */
    h1, h2, h3 {
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

# --- OTURUM DURUMU (ANIMASYON İÇİN) ---
if 'animation' not in st.session_state:
    st.session_state.animation = False
if 'frame' not in st.session_state:
    st.session_state.frame = 0

# --- YAN MENÜ (AYARLAR) ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    st.subheader("📊 Parametreler")
    
    # Enerji Slider
    energy = st.slider(
        "⚡ Parçacık Enerjisi (E)",
        min_value=0.1, max_value=2.0, value=0.8, step=0.01
    )
    
    # Bariyer Yüksekliği Slider
    barrier_height = st.slider(
        "📈 Bariyer Yüksekliği (V₀)",
        min_value=1.0, max_value=3.0, value=1.5, step=0.01
    )
    
    # Bariyer Genişliği Slider
    barrier_width = st.slider(
        "↔️ Bariyer Genişliği (L)",
        min_value=0.5, max_value=2.5, value=1.0, step=0.01
    )
    
    st.divider()
    
    # Animasyon Butonları
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Oynat" if not st.session_state.animation else "⏸️ Duraklat", use_container_width=True):
            st.session_state.animation = not st.session_state.animation
    with col2:
        if st.button("🔄 Sıfırla", use_container_width=True):
            st.session_state.frame = 0
            st.session_state.animation = False
    
    st.divider()
    
    # Hazır Senaryolar
    st.subheader("🎭 Hazır Senaryolar")
    preset = st.radio(
        "Senaryo Seç:",
        ["Özel", "Güçlü Tünelleme", "Zayıf Tünelleme", "Klasik Geçiş", "Kritik Nokta"]
    )
    
    if preset == "Güçlü Tünelleme":
        energy, barrier_height, barrier_width = 0.9, 1.0, 0.5
    elif preset == "Zayıf Tünelleme":
        energy, barrier_height, barrier_width = 0.3, 2.5, 2.0
    elif preset == "Klasik Geçiş":
        energy, barrier_height, barrier_width = 1.8, 1.2, 1.0
    elif preset == "Kritik Nokta":
        energy, barrier_height, barrier_width = 1.5, 1.5, 1.0

# --- FİZİK HESAPLAMALARI ---
def calculate_transmission(E, V0, L):
    """Tünelleme olasılığını hesaplar"""
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

# --- GRAFİK OLUŞTURMA ---
def create_plot(E, V0, L, frame=0):
    x = np.linspace(-3, L+3, 500)
    
    # Potansiyel Bariyer
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    
    # Olasılık Hesabı
    T = calculate_transmission(E, V0, L)
    
    # Grafikleri Hazırla
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Potansiyel Enerji Profili", f"Olasılık Yoğunluğu (Geçiş: %{T*100:.2f})"),
        vertical_spacing=0.15
    )
    
    # 1. Grafik: Potansiyel ve Enerji
    fig.add_trace(go.Scatter(x=x, y=V, name="Bariyer", fill='tozeroy', 
                             fillcolor='rgba(102,126,234,0.3)', line=dict(color='#667eea', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=[E]*len(x), name=f"Enerji (E={E:.2f})",
                             line=dict(color='#ff6b6b', width=2, dash='dash')), row=1, col=1)
    
    # Animasyonlu Dalga Fonksiyonu
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
        
        fig.add_trace(go.Scatter(x=x, y=wave, name="Dalga Fonksiyonu",
                                 line=dict(color='#4ecdc4', width=2), opacity=0.7), row=1, col=1)
    
    # 2. Grafik: Olasılık Yoğunluğu
    prob = np.ones_like(x)
    if E < V0:
        kappa = np.sqrt(2 * max(V0 - E, 0.01))
        prob[x < 0] = 1.0
        prob[(x >= 0) & (x <= L)] = np.exp(-2 * kappa * x[(x >= 0) & (x <= L)])
        prob[x > L] = T
    
    fig.add_trace(go.Scatter(x=x, y=prob, name="|ψ|²", fill='tozeroy',
                             fillcolor='rgba(168,85,247,0.3)', line=dict(color='#a855f7', width=2)), row=2, col=1)
    
    # Çizgiler (Bariyer Sınırları)
    for row in [1, 2]:
        fig.add_vline(x=0, line_dash="dot", line_color="gray", row=row, col=1)
        fig.add_vline(x=L, line_dash="dot", line_color="gray", row=row, col=1)
    
    fig.update_layout(height=700, showlegend=True, template="plotly_white", margin=dict(t=50, b=50))
    return fig, T

# --- ANA EKRAN DÜZENİ ---
col1, col2 = st.columns([2, 1])

with col1:
    # Animasyon Döngüsü
    if st.session_state.animation:
        st.session_state.frame += 1
        animated_energy = energy + 0.2 * np.sin(st.session_state.frame * 0.1)
        animated_energy = max(0.1, min(2.0, animated_energy))
    else:
        animated_energy = energy
    
    # Grafiği Çiz
    fig, T = create_plot(animated_energy if st.session_state.animation else energy,
                         barrier_height, barrier_width, st.session_state.frame)
    
    st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.animation:
        st.rerun()

with col2:
    st.markdown("### 📊 Sonuçlar")
    
    T_final = calculate_transmission(energy, barrier_height, barrier_width)
    
    # Sonuç Kutusu
    st.markdown(f"""
    <div style='background: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h4 style='color: #667eea; margin:0;'>Tünelleme Olasılığı</h4>
        <h1 style='color: #667eea; font-size: 3rem; margin: 10px 0;'>{T_final:.4f}</h1>
        <h3 style='color: #764ba2; margin:0;'>%{T_final*100:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if energy >= barrier_height:
        st.success("⚡ Klasik Geçiş (E ≥ V₀)")
    else:
        st.warning("🌊 Kuantum Tünelleme (E < V₀)")
    
    st.markdown("### 📈 İstatistikler")
    col_a, col_b = st.columns(2)
    col_a.metric("E/V₀ Oranı", f"{energy/barrier_height:.3f}")
    col_a.metric("Yansıma", f"{(1-T_final)*100:.1f}%")
    col_b.metric("Bariyer Alanı", f"{barrier_height * barrier_width:.2f}")

# --- ALT BİLGİ ---
st.divider()
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>TÜBİTAK 2204-A için Python & Streamlit ile hazırlanmıştır.</p>
</div>
""", unsafe_allow_html=True)
