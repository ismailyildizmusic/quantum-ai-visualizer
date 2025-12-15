"""
Quantum AI Visualizer Pro - TÜBİTAK 2204-A
Final Versiyon: Fizik Motoru + AI Yorumlayıcı + Kuantum Sonifikasyon + Canlı Animasyon
Hazırlayan: Dr. İsmail Yıldız & Gemini
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io.wavfile as wav
import io
import time

# -------------------------
# 1. Sayfa ve Stil Ayarları
# -------------------------
st.set_page_config(
    page_title="Quantum AI Visualizer Pro",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Gelişmiş CSS (Glassmorphism, Modern Kartlar ve Titreme Önleyici)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: #f8fafc;
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
    
    /* Metrik Değerleri */
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
    
    /* Buton Stili */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
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
    if abs(E - V0) < 1e-5: E += 1e-5 # Singülarite koruması

    k = np.sqrt(2 * E)
    
    if E < V0:
        # Tünelleme Rejimi
        kappa = np.sqrt(2 * (V0 - E))
        sinh_val = np.sinh(kappa * L)
        denom = 1 + (V0**2 * sinh_val**2) / (4 * E * (V0 - E))
        T = 1 / denom
    else:
        # Klasik Geçiş Rejimi
        q = np.sqrt(2 * (E - V0))
        sin_val = np.sin(q * L)
        denom = 1 + (V0**2 * sin_val**2) / (4 * E * (E - V0))
        T = 1 / denom
            
    return np.clip(T, 0.0, 1.0)

def generate_wavefunction_frame(x, E, V0, L, T, phase):
    """
    Animasyon için anlık dalga fonksiyonu hesaplar (Faz eklenmiş).
    """
    psi_real = np.zeros_like(x)
    k = np.sqrt(2*E)
    
    # Bölge 1: Gelen + Yansıyan
    mask1 = x < 0
    R = 1 - T 
    # Gelen dalga (Sağa) + Yansıyan (Sola) - Zamana bağlı faz eklendi
    psi_real[mask1] = np.cos(k * x[mask1] - phase) + np.sqrt(R) * np.cos(-k * x[mask1] - phase)
    
    # Bölge 2: Bariyer İçi
    mask2 = (x >= 0) & (x <= L)
    if E < V0:
        kappa = np.sqrt(2*(V0-E))
        decay = np.exp(-kappa * x[mask2])
        # Faz uyumu için yaklaşık ölçekleme
        scale = psi_real[x<0][-1] / decay[0] if len(decay) > 0 else 1
        # Tünelleme sırasında genlik azalır ama faz salınımı devam eder
        psi_real[mask2] = decay * scale * np.cos(-phase) 
    else:
        q = np.sqrt(2*(E-V0))
        psi_real[mask2] = np.cos(q * x[mask2] - phase) 

    # Bölge 3: Geçen
    mask3 = x > L
    if np.any(mask3):
        # Geçen dalga (sadece sağa gider)
        psi_real[mask3] = np.sqrt(T) * np.cos(k * (x[mask3]-L) - phase)
    
    return psi_real

# -------------------------
# 3. Kuantum Sonifikasyon (Ses)
# -------------------------
def generate_quantum_sound(T, E):
    sample_rate = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    base_freq = 220 + (E * 200) 
    audio = np.sin(2 * np.pi * base_freq * t)
    
    if T > 0.05: audio += 0.5 * np.sin(2 * np.pi * base_freq * 1.5 * t)
    if T > 0.3: audio += 0.25 * np.sin(2 * np.pi * base_freq * 2.0 * t)
        
    volume = 0.2 + (T * 0.8)
    audio = audio * volume
    audio = audio / np.max(np.abs(audio) + 1e-9)
    audio = (audio * 32767).astype(np.int16)
    
    virtual_file = io.BytesIO()
    wav.write(virtual_file, sample_rate, audio)
    return virtual_file

# -------------------------
# 4. Arayüz Mantığı
# -------------------------

# State Yönetimi (Animasyon durumu için)
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False

# Başlık
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;">⚛️ Quantum AI Visualizer <span style="font-size:0.5em; vertical-align:middle; background:#3b82f6; padding:4px 12px; border-radius:20px; text-transform:uppercase; letter-spacing:1px;">Pro Edition</span></h1>
    <p style="margin-top:10px; color:#cbd5e1; font-weight:300;">Yapay Zeka Destekli Kuantum Tünelleme Simülasyonu</p>
</div>
""", unsafe_allow_html=True)

# Yan Panel
with st.sidebar:
    st.header("🎛️ Parametreler")
    st.info("Enerjiyi (E), Bariyer (V₀) seviyesinin altına çekerek tünellemeyi test edin.")
    
    E = st.slider("⚡ Parçacık Enerjisi (E)", 0.1, 2.5, 0.8, 0.01)
    V0 = st.slider("🧱 Bariyer Yüksekliği (V₀)", 0.5, 3.0, 1.5, 0.01)
    L = st.slider("↔️ Bariyer Genişliği (L)", 0.5, 3.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("### 🎬 Animasyon Kontrolü")
    
    # Oynat/Durdur Butonu
    if st.button("▶️ Oynat / ⏸️ Durdur", use_container_width=True):
        st.session_state.animation_running = not st.session_state.animation_running
    
    st.caption(f"Durum: {'Çalışıyor 🟢' if st.session_state.animation_running else 'Durdu 🔴'}")
    
    st.markdown("---")
    st.markdown("### 🧬 Senaryolar")
    col_sc1, col_sc2 = st.columns(2)
    if col_sc1.button("Duvar"): E, V0, L = 0.5, 2.0, 2.0
    if col_sc2.button("Nano Sızma"): E, V0, L = 1.4, 1.5, 0.5

# Hesaplamalar
T = solve_schrodinger(E, V0, L)
x = np.linspace(-3, L+3, 800)
V_pot = np.zeros_like(x)
V_pot[(x>=0) & (x<=L)] = V0

# -------------------------
# 5. Görselleştirme (Placeholder Yöntemi)
# -------------------------
col_viz, col_res = st.columns([2, 1])

with col_viz:
    st.markdown("### 🌊 Dalga Fonksiyonu")
    
    # Grafiği içine çizeceğimiz BOŞ KUTU (Placeholder)
    # Bu teknik sayesinde sayfa titremez, sadece grafik güncellenir.
    chart_placeholder = st.empty()

    # Grafik Çizim Fonksiyonu (Tekrarlı kullanım için)
    def draw_chart(phase_val):
        psi_current = generate_wavefunction_frame(x, E, V0, L, T, phase_val)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Potansiyel Enerji ve Parçacık", "Olasılık Yoğunluğu |ψ|²"))
        
        # Grafik 1: Potansiyel ve Dalga
        fig.add_trace(go.Scatter(x=x, y=V_pot, name="Bariyer", fill='tozeroy', 
                                 line=dict(color='rgba(30, 41, 59, 0.5)', width=0),
                                 fillcolor='rgba(30, 41, 59, 0.1)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=x, y=[E]*len(x), name="Enerji", 
                                 line=dict(color='#ef4444', width=2, dash='dash')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=x, y=E + 0.4*psi_current, name="ψ(x)", 
                                 line=dict(color='#3b82f6', width=2)), row=1, col=1)

        # Grafik 2: Olasılık (Durağan olduğu için fazdan etkilenmez ama yeniden çizilmeli)
        prob = (generate_wavefunction_frame(x, E, V0, L, T, 0))**2 # Olasılık için fazsız hali kullan
        fig.add_trace(go.Scatter(x=x, y=prob, name="|ψ|²", fill='tozeroy', 
                                 line=dict(color='#8b5cf6', width=2),
                                 fillcolor='rgba(139, 92, 246, 0.2)'), row=2, col=1)

        fig.update_layout(height=600, template="plotly_white", showlegend=False,
                          margin=dict(l=20, r=20, t=40, b=20),
                          xaxis=dict(range=[-3, L+3]), # Sabit eksen (titremeyi önler)
                          yaxis=dict(range=[0, max(3.0, V0+1)])) # Sabit Y ekseni
        return fig

    # Animasyon Döngüsü
    if st.session_state.animation_running:
        # 100 karelik bir döngü yapıyoruz, sonra Streamlit tekrar başa sarar
        for i in range(50):
            phase = i * 0.2
            fig = draw_chart(phase)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.02) # Hız ayarı (daha düşük = daha hızlı)
        st.rerun() # Döngü bitince sayfayı yenile (Sürekli akış için)
    else:
        # Animasyon durduysa tek kare çiz
        fig = draw_chart(0)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

with col_res:
    # Sonuç Kartı
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b; font-size:0.9rem; font-weight:600;">GEÇİŞ OLASILIĞI (T)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">%{T*100:.3f}</div>', unsafe_allow_html=True)
    
    if E < V0:
        if T > 0.01: st.success("🌊 Kuantum Tünelleme")
        else: st.error("🧱 Tünelleme Engellendi")
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

    # AI Analizi
    st.markdown("### 🧠 AI Analizi")
    ai_text = ""
    if T < 0.0001: ai_text = "🚫 **Durum: Tam İzolasyon.** Bariyer parçacık için aşılmaz bir duvar. Klasik fizikteki 'duvara çarpma' durumu."
    elif T < 0.1: ai_text = "🌑 **Durum: Zayıf Tünelleme.** Çok nadir bir olay. Parçacıkların çoğu yansıyor, çok azı sızıyor."
    elif T < 0.9: ai_text = "🌓 **Durum: Güçlü Geçiş.** Parçacık enerjisi bariyere yakın. STM mikroskopları bu prensiple çalışır."
    else: ai_text = "🌕 **Durum: Serbest Geçiş.** Enerji bariyeri aştı. Klasik fizik kuralları baskın."
    
    st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)

    # Sonifikasyon
    st.markdown("### 🎵 Kuantum Sesi")
    audio_data = generate_quantum_sound(T, E)
    st.audio(audio_data, format='audio/wav')

# -------------------------
# Alt Bilgi
# -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.85rem;">
    TÜBİTAK 2204-A Ortaokul Öğrencileri Araştırma Projeleri Yarışması<br>
    © 2025 Dr. İsmail Yıldız
</div>
""", unsafe_allow_html=True)
