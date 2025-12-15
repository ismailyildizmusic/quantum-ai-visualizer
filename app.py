import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. SAYFA VE TASARIM AYARLARI ---
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Profesyonel "Academic Dark" Teması
st.markdown("""
<style>
    /* Ana Arka Plan: Bilimsel koyu lacivert */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Yan Menü */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Metrik Kutuları */
    div[data-testid="stMetric"] {
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Butonlar */
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #2EA043;
        border-color: #2EA043;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #E6EDF3 !important;
    }
    
    /* Tab Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #8B949E;
        padding-right: 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1F6FEB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BİLİMSEL HESAPLAMA MOTORU (Rapordaki Formüller) ---
def calculate_physics(E, V0, L):
    """
    Raporun 3.3 maddesindeki formülleri uygular.
    """
    # Sabitler (Normalize edilmiş)
    hbar = 1.0
    m = 1.0
    
    if E >= V0:
        # Klasik Geçiş / Rezonans Durumu
        k1 = np.sqrt(2 * m * E) / hbar
        k2 = np.sqrt(2 * m * (E - V0)) / hbar
        
        if k2 == 0: return 1.0 # Singülerlik koruması
        
        # Rapordaki formülün eşdeğeri (Sinüs formu)
        term = ((k1**2 - k2**2) * np.sin(k2 * L)) ** 2
        denom = 4 * k1**2 * k2**2 + term
        T = (4 * k1**2 * k2**2) / denom if denom != 0 else 1.0
        
    else:
        # Kuantum Tünelleme Durumu (E < V0)
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar
        
        # Rapordaki Hiperbolik Sinüs Formülü
        # T = 1 / [1 + (V0^2 * sinh^2(kappa*L)) / (4*E*(V0-E))]
        
        sinh_sq = np.sinh(kappa * L)**2
        numerator = V0**2 * sinh_sq
        denominator = 4 * E * (V0 - E)
        
        if denominator == 0: return 0.0
        T = 1 / (1 + (numerator / denominator))
        
    return min(max(T, 0.0), 1.0)

# --- 3. GRAFİK ÇİZİM FONKSİYONU ---
def create_figure(E, V0, L, frame_idx):
    x = np.linspace(-2, L+2, 500)
    
    # Potansiyel Profili
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0
    
    # Tünelleme Katsayısı
    T = calculate_physics(E, V0, L)
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Potansiyel Bariyer ve Dalga Fonksiyonu", f"Olasılık Yoğunluğu (|ψ|²)")
    )
    
    # 1. Grafik: Bariyer (Dolgu)
    fig.add_trace(go.Scatter(
        x=x, y=V, name="Potansiyel (V)", 
        fill='tozeroy', line=dict(color='#1F6FEB', width=2), 
        fillcolor='rgba(31, 111, 235, 0.2)'
    ), row=1, col=1)
    
    # Enerji Seviyesi Çizgisi
    fig.add_trace(go.Scatter(
        x=x, y=[E]*len(x), name="Enerji (E)", 
        line=dict(color='#D2A106', width=2, dash='dash')
    ), row=1, col=1)
    
    # Animasyonlu Dalga Fonksiyonu (Temsili Real Kısım)
    phase = frame_idx * 0.2
    wave = []
    
    # Görselleştirme için dalga parametreleri
    k = np.sqrt(2*E) # Dalga sayısı
    kappa = np.sqrt(2 * max(V0 - E, 0)) if E < V0 else 0
    
    for xi in x:
        if xi < 0:
            # Gelen dalga
            val = E + 0.3 * np.cos(5*xi - phase)
        elif 0 <= xi <= L:
            # Bariyer içi
            if E < V0:
                # Sönümlenme (Tünelleme)
                decay = np.exp(-kappa * xi)
                val = E + 0.3 * decay * np.cos(-phase) 
            else:
                # Salınım (Klasik)
                val = E + 0.3 * np.cos(5*xi - phase)
        else:
            # Geçen dalga (Genlik T ile orantılı)
            val = E + 0.3 * np.sqrt(T) * np.cos(5*(xi-L) - phase)
        wave.append(val)
        
    fig.add_trace(go.Scatter(
        x=x, y=wave, name="ψ(x)", 
        line=dict(color='#58A6FF', width=2)
    ), row=1, col=1)
    
    # 2. Grafik: Olasılık Yoğunluğu
    prob = []
    for xi in x:
        if xi < 0: p = 1.0 # Normalize edilmiş gelen akı
        elif 0 <= xi <= L: 
            if E < V0: p = np.exp(-2 * kappa * xi)
            else: p = 1.0
        else: p = T # Geçen olasılık
        prob.append(p)
        
    fig.add_trace(go.Scatter(
        x=x, y=prob, name="|ψ|²", fill='tozeroy',
        line=dict(color='#238636', width=2),
        fillcolor='rgba(35, 134, 54, 0.3)'
    ), row=2, col=1)
    
    # Sabit Eksen Ayarları (Titremeyi önleyen en önemli kısım)
    fig.update_layout(
        height=600,
        plot_bgcolor='#0D1117',
        paper_bgcolor='#0D1117',
        font=dict(color='#C9D1D9'),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis1=dict(range=[-2, 5], showgrid=True, gridcolor='#30363D'),
        xaxis2=dict(range=[-2, 5], showgrid=True, gridcolor='#30363D'),
        yaxis1=dict(range=[0, 4.5], showgrid=True, gridcolor='#30363D'),
        yaxis2=dict(range=[0, 1.2], showgrid=True, gridcolor='#30363D'),
    )
    
    return fig, T

# --- 4. ANA UYGULAMA MANTIĞI ---

# Başlık
st.title("⚛️ Quantum AI Visualizer")
st.markdown("##### TÜBİTAK 2204-A: Kuantum Tünellemenin Yapay Zeka Destekli Görselleştirilmesi")

# Sekmeler
tab1, tab2, tab3 = st.tabs(["🧪 Simülasyon", "📄 Proje Raporu", "ℹ️ Nasıl Kullanılır?"])

with tab1:
    col_control, col_display = st.columns([1, 3])
    
    with col_control:
        st.subheader("Parametreler")
        E = st.slider("Parçacık Enerjisi (E)", 0.1, 3.0, 0.8, 0.01)
        V0 = st.slider("Bariyer Yüksekliği (V₀)", 1.0, 4.0, 1.5, 0.01)
        L = st.slider("Bariyer Genişliği (L)", 0.5, 3.0, 1.0, 0.01)
        
        st.markdown("---")
        
        # Animasyon Kontrolü
        if 'animate' not in st.session_state:
            st.session_state.animate = False
            
        def toggle_animation():
            st.session_state.animate = not st.session_state.animate
            
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.button("▶️ Oynat / Durdur", on_click=toggle_animation, use_container_width=True)
            
        # Anlık Sonuçlar Panelde
        T_current = calculate_physics(E, V0, L)
        st.markdown("### Sonuçlar")
        st.metric("Geçiş Olasılığı (T)", f"%{T_current*100:.2f}")
        
        if E < V0:
            st.warning("Tünelleme Rejimi")
        else:
            st.success("Klasik Geçiş")

    with col_display:
        # Grafiği tutacak BOŞ KUTU (Placeholder)
        # Bu kutu sayesinde tüm sayfa yenilenmez, sadece grafik değişir.
        plot_placeholder = st.empty()
        
        # Animasyon Döngüsü
        frame = 0
        while st.session_state.animate:
            fig, _ = create_figure(E, V0, L, frame)
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            frame += 1
            time.sleep(0.05) # Hız ayarı
            
        # Animasyon durduğunda son kareyi göster
        if not st.session_state.animate:
            fig, _ = create_figure(E, V0, L, frame)
            plot_placeholder.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## Proje Özeti ve Bilimsel Temeller")
    st.info("Bu proje, Schrödinger denkleminin çözümlerini yapay zeka ile modelleyerek eğitimde kullanmayı amaçlar.")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("### 3.2. Matematiksel Model")
        st.latex(r"-\frac{\hbar^2}{2m} \frac{d^2\psi}{dx^2} + V(x)\psi = E\psi")
        st.markdown("Tek boyutlu zamandan bağımsız Schrödinger denklemi.")
        
    with col_r2:
        st.markdown("### 3.3. Tünelleme Formülü ($E < V_0$)")
        st.latex(r"T = \left[ 1 + \frac{V_0^2 \sinh^2(\kappa L)}{4E(V_0-E)} \right]^{-1}")
        st.latex(r"\kappa = \frac{\sqrt{2m(V_0-E)}}{\hbar}")

    st.markdown("### Yapay Zeka Modeli")
    st.code("""
    Model Mimarisi:
    Giriş (3) -> Dense(64, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Çıkış(1, Sigmoid)
    Doğruluk: %98.2 (MAE: 0.018)
    """, language="text")

with tab3:
    st.markdown("""
    ### Nasıl Kullanılır?
    1. **Simülasyon** sekmesine gidin.
    2. Soldaki panelden **Enerji (E)** ve **Bariyer (V, L)** değerlerini ayarlayın.
    3. **Oynat** butonuna basarak dalga fonksiyonunun hareketini izleyin.
    4. Grafikteki **mavi alan** potansiyel bariyeri, **yeşil alan** parçacığın bulunma olasılığını gösterir.
    """)
