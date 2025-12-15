"""
Quantum AI Visualizer Pro - TÜBİTAK 2204-A
Geliştirilmiş Versiyon: Fizik Motoru + AI Yorumlayıcı + Kuantum Sonifikasyon
Hazırlayan: Dr. İsmail Yıldız & Gemini
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io.wavfile as wav
import io

# -------------------------
# 1. Sayfa ve Stil Ayarları
# -------------------------
st.set_page_config(
    page_title="Quantum AI Visualizer Pro",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Gelişmiş CSS (Glassmorphism ve Modern Kartlar)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: #f8fafc; /* Slate-50 - Temiz Beyaz/Gri */
    }
    
    /* Üst Başlık Kartı */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid #334155;
    }
    
    /* Bilgi Kartları */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrik Değerleri (Renkli Sayılar) */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Yapay Zeka Kutusu */
    .ai-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #1e293b;
    }
    
    /* Plotly Arka Planını Temizle */
    .js-plotly-plot .plotly .main-svg {
        background: rgba(0,0,0,0) !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 2. Fizik Motoru (Schrödinger Çözücü)
# -------------------------
def solve_schrodinger(E, V0, L):
    """
    Analitik Çözüm: 1D Dikdörtgen Bariyer
    ħ=1, m=1 birim sisteminde.
    """
    # Sıfıra bölme hatasını önlemek için E ve V0 eşitliğini engelle
    if abs(E - V0) < 1e-5:
        E += 1e-5

    k = np.sqrt(2 * E)
    
    if E < V0:
        # Tünelleme Rejimi (E < V0)
        kappa = np.sqrt(2 * (V0 - E))
        # Hiperbolik sinüs formülü (Proje raporundaki formül)
        sinh_val = np.sinh(kappa * L)
        denom = 1 + (V0**2 * sinh_val**2) / (4 * E * (V0 - E))
        T = 1 / denom
    else:
        # Klasik Geçiş Rejimi (E > V0)
        q = np.sqrt(2 * (E - V0))
        # Trigonometrik sinüs formülü
        sin_val = np.sin(q * L)
        denom = 1 + (V0**2 * sin_val**2) / (4 * E * (E - V0))
        T = 1 / denom
            
    return np.clip(T, 0.0, 1.0)

def generate_wavefunction_snapshot(x, E, V0, L, T):
    """
    Görselleştirme için dalga fonksiyonunun 'Reel' kısmının anlık görüntüsü.
    """
    psi_real = np.zeros_like(x)
    k = np.sqrt(2*E)
    
    # Bölge 1: Gelen + Yansıyan
    mask1 = x < 0
    R = 1 - T 
    # Duran dalga deseni oluşur (Gelen + Yansıyan girişimi)
    psi_real[mask1] = np.cos(k * x[mask1]) + np.sqrt(R) * np.cos(-k * x[mask1])
    
    # Bölge 2: Bariyer İçi
    mask2 = (x >= 0) & (x <= L)
    if E < V0:
        kappa = np.sqrt(2*(V0-E))
        # Sönümlenen üstel fonksiyon (Exponential Decay)
        # Süreklilik için genlik ölçeklemesi
        boundary_val = psi_real[x<0][-1] if np.any(x<0) else 1.0
        decay = np.exp(-kappa * x[mask2])
        # Ölçekleme (Görsel devamlılık için yaklaşık)
        scale = boundary_val / decay[0] if len(decay) > 0 else 1
        psi_real[mask2] = decay * scale
    else:
        q = np.sqrt(2*(E-V0))
        psi_real[mask2] = np.cos(q * x[mask2]) 

    # Bölge 3: Geçen
    mask3 = x > L
    # Geçen dalga genliği T'ye bağlı
    if np.any(mask3):
        psi_real[mask3] = np.sqrt(T) * np.cos(k * (x[mask3]-L))
    
    return psi_real

# -------------------------
# 3. Kuantum Sonifikasyon (Sese Dönüştürme)
# -------------------------
def generate_quantum_sound(T, E):
    """
    Tünelleme verisini sese dönüştürür.
    Scipy kütüphanesi gerektirir.
    """
    sample_rate = 44100
    duration = 2.0 # saniye
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Enerji -> Frekans (Pitch)
    # Düşük enerji kalın ses, Yüksek enerji tiz ses
    base_freq = 220 + (E * 200) 
    
    # Tünelleme Olasılığı -> Sesin "Rengi" ve Seviyesi
    audio = np.sin(2 * np.pi * base_freq * t) # Saf ton
    
    # Tünelleme arttıkça ses zenginleşir (Harmonikler eklenir)
    if T > 0.05:
        audio += 0.5 * np.sin(2 * np.pi * base_freq * 1.5 * t) # 5'li
    if T > 0.3:
        audio += 0.25 * np.sin(2 * np.pi * base_freq * 2.0 * t) # Oktav
        
    # Ses seviyesi (Volume) T'ye bağlı
    # T çok düşükse ses çok kısık olur
    volume = 0.2 + (T * 0.8)
    audio = audio * volume
    
    # 16-bit WAV formatına çevir
    audio = audio / np.max(np.abs(audio) + 1e-9) # Normalize
    audio = (audio * 32767).astype(np.int16)
    
    virtual_file = io.BytesIO()
    wav.write(virtual_file, sample_rate, audio)
    return virtual_file

# -------------------------
# 4. Arayüz Tasarımı
# -------------------------

# Başlık Kartı
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">⚛️ Quantum AI Visualizer <span style="font-size:0.5em; vertical-align:middle; background:#3b82f6; padding:4px 12px; border-radius:20px; text-transform:uppercase; letter-spacing:1px;">Pro Edition</span></h1>
    <p style="margin-top:10px; color:#cbd5e1; font-weight:300;">Yapay Zeka Destekli Kuantum Tünelleme Simülasyonu ve Sonifikasyon</p>
</div>
""", unsafe_allow_html=True)

# Yan Panel (Sidebar)
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.header("🎛️ Parametreler")
    st.info("💡 **İpucu:** Enerjiyi (E) bariyer yüksekliğinin (V₀) altına düşürerek tünellemeyi gözlemleyin.")
    
    E = st.slider("⚡ Parçacık Enerjisi (E)", 0.1, 2.5, 0.8, 0.01)
    V0 = st.slider("🧱 Bariyer Yüksekliği (V₀)", 0.5, 3.0, 1.5, 0.01)
    L = st.slider("↔️ Bariyer Genişliği (L)", 0.5, 3.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("### 🧬 Hazır Senaryolar")
    
    col_sc1, col_sc2 = st.columns(2)
    if col_sc1.button("Duvar"):
        E, V0, L = 0.5, 2.0, 2.0
    if col_sc2.button("Nano Sızma"):
        E, V0, L = 1.4, 1.5, 0.5

# Hesaplamalar
T = solve_schrodinger(E, V0, L)
x = np.linspace(-3, L+3, 800)
psi = generate_wavefunction_snapshot(x, E, V0, L, T)
V_pot = np.zeros_like(x)
V_pot[(x>=0) & (x<=L)] = V0

# -------------------------
# 5. Görselleştirme ve Sonuçlar
# -------------------------
col_viz, col_res = st.columns([2, 1])

with col_viz:
    st.markdown("### 🌊 Dalga Fonksiyonu ve Potansiyel")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1,
                        subplot_titles=("Potansiyel Enerji ve Parçacık", "Olasılık Yoğunluğu |ψ|²"))
    
    # Grafik 1: Potansiyel ve Re(psi)
    # Bariyer (Gri alan)
    fig.add_trace(go.Scatter(x=x, y=V_pot, name="Bariyer V(x)", 
                             fill='tozeroy', line=dict(color='rgba(30, 41, 59, 0.5)', width=0),
                             fillcolor='rgba(30, 41, 59, 0.1)'), row=1, col=1)
    
    # Enerji Seviyesi (Kırmızı Çizgi)
    fig.add_trace(go.Scatter(x=x, y=[E]*len(x), name="Enerji E", 
                             line=dict(color='#ef4444', width=2, dash='dash')), row=1, col=1)
    
    # Dalga Fonksiyonu (Mavi)
    fig.add_trace(go.Scatter(x=x, y=E + 0.4*psi, name="ψ(x) (Reel)", 
                             line=dict(color='#3b82f6', width=2)), row=1, col=1)

    # Grafik 2: Olasılık (Mor)
    prob = psi**2
    fig.add_trace(go.Scatter(x=x, y=prob, name="|ψ|²", 
                             fill='tozeroy', line=dict(color='#8b5cf6', width=2),
                             fillcolor='rgba(139, 92, 246, 0.2)'), row=2, col=1)

    # Eksen Süslemeleri
    fig.update_layout(height=600, template="plotly_white", 
                      hovermode="x unified",
                      margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation="h", y=1.1))
    
    st.plotly_chart(fig, use_container_width=True)

with col_res:
    # 1. Sayısal Sonuçlar Kartı
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b; font-size:0.9rem; font-weight:600;">GEÇİŞ OLASILIĞI (T)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">%{T*100:.3f}</div>', unsafe_allow_html=True)
    
    if E < V0:
        if T > 0.01:
             st.success("🌊 Kuantum Tünelleme Başarılı")
        else:
             st.error("🧱 Tünelleme Engellendi")
    else:
        st.info("🚀 Klasik Geçiş")
        
    st.markdown(f"""
    <div style="margin-top:15px; font-size:0.9rem; border-top:1px solid #eee; padding-top:10px;">
    <b>Detaylı Veriler:</b><br>
    • E / V₀ Oranı: <b>{E/V0:.2f}</b><br>
    • Yansıma (R): <b>%{(1-T)*100:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Yapay Zeka Analizi (Expert System)
    st.markdown("### 🧠 AI Analizi")
    
    ai_text = ""
    if T < 0.0001:
        ai_text = "🚫 **Durum:** Tam İzolasyon. Bariyer parçacık için aşılmaz bir duvar gibi davranıyor. Klasik fizikteki 'topun duvardan sekmesi' durumu geçerli."
    elif T < 0.1:
        ai_text = "🌑 **Durum:** Zayıf Tünelleme. Parçacıkların büyük çoğunluğu yansıyor. Ancak kuantum belirsizliği sayesinde çok az bir kısmı 'hayalet gibi' karşıya geçiyor."
    elif T < 0.9:
        ai_text = "🌓 **Durum:** Güçlü Tünelleme/Geçiş. Parçacık enerjisi bariyerle yarışıyor. Bariyerin varlığı hissediliyor ancak geçiş yüksek oranda mümkün. STM mikroskopları bu aralıkta çalışır."
    else:
        ai_text = "🌕 **Durum:** Şeffaf Bariyer. Enerji bariyeri aştı. Parçacık neredeyse hiç engel yokmuş gibi ilerliyor. Klasik fizik kuralları baskın."

    st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)

    # 3. Kuantum Sonifikasyon
    st.markdown("### 🎵 Sonifikasyon")
    st.caption("Veriyi sese dönüştürerek tünellemeyi 'duyun'.")
    
    audio_data = generate_quantum_sound(T, E)
    st.audio(audio_data, format='audio/wav')

# -------------------------
# Alt Bilgi
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.85rem;">
    TÜBİTAK 2204-A Ortaokul Öğrencileri Araştırma Projeleri Yarışması<br>
    © 2025 Dr. İsmail Yıldız | Python, Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
