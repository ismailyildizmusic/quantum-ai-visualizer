"""
Quantum AI Visualizer - Pro Version
Titreme Engelleyici (Anti-Flicker) Modu Aktif
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SAYFA YAPILANDIRMASI (EN ÜSTTE) ---
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS İLE TİTREME ENGELLEME VE TASARIM ---
st.markdown("""
<style>
    /* 1. TİTREME ENGELLEYİCİ KODLAR (Anti-Shake) */
    /* Kaydırma çubuğunu her zaman göster (Sağa sola oynamayı engeller) */
    html {
        overflow-y: scroll;
    }
    
    /* Üstteki 'Running' animasyonunu ve header'ı gizle (Tepedeki oynamayı engeller) */
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Sayfa yenilenirken beyaz flaşı engelle */
    .stApp {
        background: linear-gradient(135deg, #141E30 0%, #243B55 100%);
        background-attachment: fixed; /* Arka planı çivile */
    }

    /* 2. GÖRSEL KALİTE (Glassmorphism) */
    /* Metrik Kutuları */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-color: #00d2ff;
    }
    
    /* Yazı Tipleri ve Renkler */
    h1, h2, h3, h4, p, span, li {
        color: #E0E0E0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Yan Menü Güzelleştirme */
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 30, 48, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buton Tasarımları */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #141E30;
        border: none;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. BAŞLIK VE GİRİŞ ---
st.markdown("""
<div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 15px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.1);'>
    <h1 style='font-size: 3rem; background: -webkit-linear-gradient(#eee, #333); -webkit-background-clip: text; text-shadow: 0px 0px 20px rgba(0, 210, 255, 0.5);'>
        ⚛️ Quantum AI Visualizer
    </h1>
    <p style='font-size: 1.2rem; letter-spacing: 1px; opacity: 0.8;'>
        Yapay Zeka Destekli Kuantum Tünelleme Laboratuvarı
    </p>
    <div style='margin-top: 10px;'>
        <span style='background: #00d2ff; color: #000 !important; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;'>TÜBİTAK 2204-A</span>
        <span style='background: #92FE9D; color: #000 !important; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; margin-left: 10px;'>Pro Version</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 4. AYARLAR VE STATE YÖNETİMİ ---
if 'run_anim' not in st.session_state:
    st.session_state.run_anim = False
if 'frame_idx' not in st.session_state:
    st.session_state.frame_idx = 0

# Yan Menü
with st.sidebar:
    st.header("🎛️ Kontrol Paneli")
    
    st.info("Parametreleri buradan değiştirin:")
    energy = st.slider("⚡ Enerji (E)", 0.1, 3.0, 0.8, 0.01, help="Parçacığın sahip olduğu toplam enerji")
    barrier_height = st.slider("🧱 Bariyer Yüksekliği (V₀)", 1.0, 4.0, 1.5, 0.01, help="Aşılması gereken potansiyel engel")
    barrier_width = st.slider("↔️ Bariyer Genişliği (L)", 0.5, 3.0, 1.0, 0.01, help="Engelin kalınlığı")

    st.markdown("---")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        # Animasyon Butonu
        btn_label = "⏸️ Durdur" if st.session_state.run_anim else "▶️ Oynat"
        if st.button(btn_label):
            st.session_state.run_anim = not st.session_state.run_anim
            st.rerun()
            
    with col_b2:
        if st.button("🔄 Sıfırla"):
            st.session_state.run_anim = False
            st.session_state.frame_idx = 0
            st.rerun()

    st.markdown("---")
    st.caption("Geliştirici: Dr. İsmail Yıldız")

# --- 5. FİZİK MOTORU ---
def solve_schrodinger(E, V0, L):
    """Tünelleme olasılığını hesaplayan fizik motoru"""
    if E >= V0: # Klasik Geçiş
        k1 = np.sqrt(2 * E)
        k2 = np.sqrt(2 * (E - V0))
        denom = (k1 + k2)**2 - (k1 - k2)**2 * np.sin(k2 * L)**2
        T = (4 * k1 * k2 / denom) if denom != 0 else 1.0
    else: # Kuantum Tünelleme
        kappa = np.sqrt(2 * (V0 - E))
        if kappa * L > 50: return 0.0 # Taşma koruması
        sinh_sq = np.sinh(kappa * L)**2
        denom = 4 * E * (V0 - E)
        T = (1 / (1 + (V0**2 * sinh_sq) / denom)) if denom != 0 else 0.0
    return T

# --- 6. GRAFİK OLUŞTURUCU ---
def draw_scene(E, V0, L, frame):
    x = np.linspace(-3, L+3, 600)
    
    # Potansiyel Profili
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    
    T = solve_schrodinger(E, V0, L)
    
    # Grafik İskeleti
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Potansiyel Enerji ve Dalga Fonksiyonu", "Olasılık Yoğunluğu (|ψ|²)")
    )
    
    # 1. Grafik: Bariyer ve Enerji
    fig.add_trace(go.Scatter(x=x, y=V, name="Bariyer", fill='tozeroy', 
                             line=dict(color='#00d2ff', width=1), fillcolor='rgba(0, 210, 255, 0.15)'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=[E]*len(x), name="Enerji", 
                             line=dict(color='#ff6b6b', width=2, dash='dash')), row=1, col=1)

    # Animasyonlu Dalga
    phase = frame * 0.25
    wave_real = []
    
    # Hızlı hesaplama için döngü optimizasyonu
    k_inc = np.sqrt(2 * E)
    kappa = np.sqrt(2 * max(V0 - E, 0.0))
    
    for xi in x:
        if xi < 0: # Gelen Bölge
            val = E + 0.4 * np.cos(5 * xi - phase)
        elif 0 <= xi <= L: # Bariyer İçi
            if E < V0: # Tünelleme
                decay = np.exp(-kappa * xi)
                val = E + 0.4 * decay * np.cos(-phase) 
            else: # Klasik Geçiş
                val = E + 0.4 * np.cos(5 * xi - phase)
        else: # Geçen Bölge
            val = E + 0.4 * np.sqrt(T) * np.cos(5 * (xi - L) - phase)
        wave_real.append(val)
        
    fig.add_trace(go.Scatter(x=x, y=wave_real, name="Dalga (Re)", 
                             line=dict(color='#ffffff', width=2), opacity=0.9), row=1, col=1)

    # 2. Grafik: Olasılık
    prob = []
    for xi in x:
        if xi < 0: p = 1.0 + 0.1 * np.sin(phase/2) # Girişim efekti
        elif 0 <= xi <= L: 
            if E < V0: p = np.exp(-2 * kappa * xi)
            else: p = 1.0 # Basitleştirilmiş
        else: p = T
        prob.append(p)

    fig.add_trace(go.Scatter(x=x, y=prob, name="|ψ|²", fill='tozeroy',
                             line=dict(color='#92FE9D', width=2), fillcolor='rgba(146, 254, 157, 0.2)'), row=2, col=1)

    # Düzen Ayarları (Titremeyi önlemek için sabit aralıklar)
    fig.update_layout(
        height=650, # Sabit yükseklik
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis1=dict(range=[-3, 6], showgrid=True, gridcolor='rgba(255,255,255,0.1)'), # X ekseni sabit
        xaxis2=dict(range=[-3, 6], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis1=dict(range=[0, 4.5], showgrid=True, gridcolor='rgba(255,255,255,0.1)'), # Y ekseni sabit (Çok önemli!)
        yaxis2=dict(range=[0, 1.2], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    )
    return fig, T

# --- 7. ANA DÖNGÜ VE GÖSTERİM ---
# Layout'u ikiye böl
col_main, col_data = st.columns([3, 1])

# Ana Grafik Alanı
with col_main:
    # Boş bir kutu oluşturup grafiği HEP buraya çizeceğiz
    # Bu sayede sayfa yapısı bozulmaz
    chart_container = st.empty()
    
    # Animasyon mantığı
    if st.session_state.run_anim:
        st.session_state.frame_idx += 1
        # Enerjide hafif salınım efekti
        dynamic_E = energy + 0.05 * np.sin(st.session_state.frame_idx * 0.1)
        fig, T_val = draw_scene(dynamic_E, barrier_height, barrier_width, st.session_state.frame_idx)
        
        # Grafiği bas
        chart_container.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
        
        # Hız ayarı (CPU'yu yormamak için)
        time.sleep(0.03)
        st.rerun()
    else:
        # Durmuş hali
        fig, T_val = draw_scene(energy, barrier_height, barrier_width, st.session_state.frame_idx)
        chart_container.plotly_chart(fig, use_container_width=True)

# Veri Paneli (Sağ Taraf)
with col_data:
    st.markdown("### 📊 Sonuçlar")
    
    T_exact = solve_schrodinger(energy, barrier_height, barrier_width)
    
    # Büyük Olasılık Göstergesi
    st.markdown(f"""
    <div style='background: rgba(0,210,255,0.1); padding: 20px; border-radius: 15px; border: 1px solid #00d2ff; text-align: center; margin-bottom: 20px;'>
        <div style='font-size: 0.9rem; color: #00d2ff;'>GEÇİŞ OLASILIĞI</div>
        <div style='font-size: 2.8rem; font-weight: bold; color: white;'>%{T_exact*100:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Yansıma Olasılığı (R)", f"%{(1-T_exact)*100:.2f}")
    st.metric("E / V₀ Oranı", f"{energy/barrier_height:.3f}")
    
    st.divider()
    
    if energy < barrier_height:
        st.warning("⚠️ TÜNELLEME REJİMİ")
        st.caption("Parçacık enerjisi bariyerden düşük olmasına rağmen 'sızarak' karşıya geçiyor.")
    else:
        st.success("✅ KLASİK GEÇİŞ")
        st.caption("Parçacık enerjisi bariyeri aşacak kadar yüksek.")

# --- 8. ALT BİLGİ ---
st.markdown("---")
with st.expander("📚 Kuantum Formülleri ve Detaylar"):
    st.latex(r"T = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)}}")
    st.markdown("TÜBİTAK 2204-A Projesi kapsamında hazırlanmıştır.")
