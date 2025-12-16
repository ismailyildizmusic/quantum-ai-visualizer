"""
Quantum AI Visualizer - TÜBİTAK 2204-A
Final Versiyon: Fizik Motoru + AI Yorumlayıcı + Kuantum Sonifikasyon + Canlı Animasyon

"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import wave


# -------------------------
# 1. Sayfa ve Stil Ayarları
# -------------------------
st.set_page_config(
    page_title="Quantum AI Visualizer Pro",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Daha sakin/profesyonel CSS (flicker yok: rerun yok)
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f8fafc;
    color: #0f172a;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 1.7rem 1.6rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    border: 1px solid rgba(148,163,184,0.25);
}
.card {
    background: white;
    padding: 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(15,23,42,0.10);
    box-shadow: 0 6px 16px rgba(15,23,42,0.06);
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 900;
    background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.1rem 0 0.3rem 0;
}
.ai-box {
    background-color: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 1.0rem;
    border-radius: 10px;
    font-size: 0.95rem;
    color: #1e293b;
}
.small {
    color: rgba(15,23,42,0.65);
    font-size: 0.92rem;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
  <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;">
    <div>
      <h1 style="margin:0; font-weight:900;">
        ⚛️ Quantum AI Visualizer
        <span style="font-size:0.45em; vertical-align:middle; background:#3b82f6; padding:4px 12px; border-radius:999px; text-transform:uppercase; letter-spacing:0.6px;">
          Pro Edition
        </span>
      </h1>
      <p style="margin-top:10px; color:#cbd5e1; font-weight:400;">
        Yapay Zeka Destekli Kuantum Tünelleme Simülasyonu • Fizik Motoru + Sonifikasyon + Animasyon
      </p>
      <div style="margin-top:10px; display:inline-block; padding:6px 12px; border-radius:999px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.18);">
        🏆 TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması
      </div>
    </div>
    <div style="color:#cbd5e1; font-size:0.9rem; text-align:right;">
      Birim sistemi: ħ = 1, m = 1 (normalize)<br/>
      Canlı animasyon: Plotly (sayfa yenilenmez)
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# -------------------------
# 2. Fizik Motoru (Sınır Koşulları ile Schrödinger Çözümü)
# -------------------------
def solve_scattering_coeffs(E: float, V0: float, L: float):
    """
    1D dikdörtgen bariyer saçılması.
    Bölge I (x<0):     ψ = e^{ikx} + r e^{-ikx}
    Bölge II (0<x<L):  ψ = A e^{iqx} + B e^{-iqx}
    Bölge III (x>L):   ψ = t e^{ikx}

    Doğal birimler: ħ=1, m=1
    k = sqrt(2E)
    E<V0 ise q = iκ
    Sol ve sağ potansiyel aynı (0) olduğundan T = |t|^2
    """
    E = float(max(E, 1e-6))
    V0 = float(max(V0, 0.0))
    L = float(max(L, 1e-6))

    # E ~ V0 singülariteye karşı küçük kaydırma
    if abs(E - V0) < 1e-6:
        E = E + 1e-6

    k = np.sqrt(2.0 * E)

    if E < V0:
        kappa = np.sqrt(2.0 * (V0 - E))
        q = 1j * kappa
    else:
        q = np.sqrt(2.0 * (E - V0))

    e_qL = np.exp(1j * q * L)
    e_mqL = np.exp(-1j * q * L)
    e_kL = np.exp(1j * k * L)

    # bilinmeyenler: [r, A, B, t]
    M = np.zeros((4, 4), dtype=np.complex128)
    b = np.zeros((4,), dtype=np.complex128)

    # x=0: 1+r = A+B
    M[0, 0] = 1.0
    M[0, 1] = -1.0
    M[0, 2] = -1.0
    b[0] = -1.0

    # x=0: ik(1-r) = i q (A-B)
    M[1, 0] = -1j * k
    M[1, 1] = -1j * q
    M[1, 2] = +1j * q
    b[1] = -1j * k

    # x=L: A e^{iqL} + B e^{-iqL} = t e^{ikL}
    M[2, 1] = e_qL
    M[2, 2] = e_mqL
    M[2, 3] = -e_kL

    # x=L: i q (A e^{iqL} - B e^{-iqL}) = i k t e^{ikL}
    M[3, 1] = 1j * q * e_qL
    M[3, 2] = -1j * q * e_mqL
    M[3, 3] = -1j * k * e_kL

    r, A, B, t = np.linalg.solve(M, b)

    T = float(np.clip(np.abs(t) ** 2, 0.0, 1.0))
    return r, A, B, t, T, k, q


def psi_x(x: np.ndarray, E: float, V0: float, L: float, phase: float):
    """ψ(x) hesapla, global faz ile Re(ψ) animasyon üret."""
    r, A, B, t, T, k, q = solve_scattering_coeffs(E, V0, L)

    psi = np.zeros_like(x, dtype=np.complex128)

    m1 = x < 0
    psi[m1] = np.exp(1j * k * x[m1]) + r * np.exp(-1j * k * x[m1])

    m2 = (x >= 0) & (x <= L)
    psi[m2] = A * np.exp(1j * q * x[m2]) + B * np.exp(-1j * q * x[m2])

    m3 = x > L
    psi[m3] = t * np.exp(1j * k * x[m3])

    psi *= np.exp(-1j * phase)
    return psi


# -------------------------
# 3. Kuantum Sonifikasyon (SciPy yok, standart WAV)
# -------------------------
@st.cache_data(show_spinner=False)
def generate_quantum_sound(T: float, E: float, duration: float = 2.0, sample_rate: int = 44100) -> bytes:
    """
    T ve E değerine göre kısa WAV üretir (bytes).
    - Temel frekans E ile artar
    - Harmonik zenginlik T ile artar
    - Ses seviyesi T ile artar
    """
    T = float(np.clip(T, 0.0, 1.0))
    E = float(max(E, 0.0))

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    base_freq = 220.0 + (E * 200.0)
    audio = np.sin(2 * np.pi * base_freq * t)

    if T > 0.05:
        audio += 0.50 * np.sin(2 * np.pi * base_freq * 1.5 * t)
    if T > 0.30:
        audio += 0.25 * np.sin(2 * np.pi * base_freq * 2.0 * t)

    volume = 0.15 + (T * 0.85)
    audio *= volume

    # normalize and convert to int16
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    audio_i16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

    return buf.getvalue()


# -------------------------
# 4. Arayüz
# -------------------------
with st.sidebar:
    st.header("🎛️ Parametreler")
    st.info("E değerini V₀ seviyesinin altına çekerek tünellemeyi gözlemleyin.")

    E = st.slider("⚡ Parçacık Enerjisi (E)", 0.10, 2.50, 0.80, 0.01)
    V0 = st.slider("🧱 Bariyer Yüksekliği (V₀)", 0.50, 3.00, 1.50, 0.01)
    L = st.slider("↔️ Bariyer Genişliği (L)", 0.50, 3.00, 1.00, 0.05)

    st.divider()
    st.markdown("### 🎬 Animasyon")
    animate = st.toggle("Re(ψ) Animasyonu", value=True)
    n_frames = st.slider("Kare sayısı", 20, 120, 60, 5)

    st.divider()
    st.markdown("### 🧬 Senaryolar")
    c1, c2 = st.columns(2)
    if c1.button("🧱 Duvar"):
        E, V0, L = 0.50, 2.00, 2.00
    if c2.button("🧬 Nano Sızma"):
        E, V0, L = 1.40, 1.50, 0.50


# Hesaplar
r, A, B, tcoef, T, k, q = solve_scattering_coeffs(E, V0, L)

x = np.linspace(-3.0, L + 3.0, 900)
V_pot = np.zeros_like(x)
V_pot[(x >= 0) & (x <= L)] = V0

psi0 = psi_x(x, E, V0, L, phase=0.0)
prob = np.abs(psi0) ** 2
# Görsel için normalize (okunabilirlik)
prob_plot = prob / (np.max(prob) + 1e-9)

re0 = np.real(psi0)
re_norm = re0 / (np.max(np.abs(re0)) + 1e-9)
scale = 0.35 * max(1.0, V0)
psi_line0 = E + scale * re_norm


# -------------------------
# 5. Görselleştirme (Plotly Frames → Sayfa yenilenmez)
# -------------------------
col_viz, col_res = st.columns([2, 1])

with col_viz:
    st.markdown("### 🌊 Dalga Fonksiyonu ve Olasılık")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Potansiyel Enerji, Enerji Seviyesi ve Re(ψ) (animasyon)", "Olasılık Yoğunluğu |ψ|² (normalize)"),
    )

    # Üst: Bariyer
    fig.add_trace(
        go.Scatter(
            x=x,
            y=V_pot,
            name="Bariyer V(x)",
            fill="tozeroy",
            line=dict(width=0),
            fillcolor="rgba(30,41,59,0.10)",
        ),
        row=1,
        col=1,
    )

    # Üst: Enerji
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[E] * len(x),
            name="Enerji E",
            line=dict(color="#ef4444", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Üst: Re(ψ) çizgisi (animasyonla güncellenecek trace)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=psi_line0,
            name="Re(ψ) (ölçekli)",
            line=dict(color="#2563eb", width=2),
        ),
        row=1,
        col=1,
    )

    # Alt: |ψ|²
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prob_plot,
            name="|ψ|²",
            fill="tozeroy",
            line=dict(color="#7c3aed", width=2),
            fillcolor="rgba(124,58,237,0.16)",
        ),
        row=2,
        col=1,
    )

    # Bariyer sınırları
    for rr in [1, 2]:
        fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="gray", row=rr, col=1)
        fig.add_vline(x=L, line_dash="dot", line_width=1, line_color="gray", row=rr, col=1)

    # Sabit eksenler (titreme hissi azalır)
    fig.update_xaxes(range=[-3, L + 3], title_text="Konum x", row=2, col=1)
    fig.update_yaxes(range=[0, max(3.0, V0 + 1.0)], title_text="Enerji / Potansiyel", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], title_text="|ψ|² (normalize)", row=2, col=1)

    fig.update_layout(
        height=680,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        showlegend=False,
    )

    # Animasyon frames (sayfa yenilenmeden)
    if animate:
        phases = np.linspace(0.0, 2 * np.pi, int(n_frames), endpoint=False)
        frames = []

        for ph in phases:
            psi = psi_x(x, E, V0, L, phase=float(ph))
            re = np.real(psi)
            re_norm_f = re / (np.max(np.abs(re)) + 1e-9)
            psi_line = E + scale * re_norm_f

            # trace index 2 (üstteki Re(ψ) trace)
            frames.append(go.Frame(data=[go.Scatter(x=x, y=psi_line)], traces=[2], name=f"{ph:.3f}"))

        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=1.13,
                    buttons=[
                        dict(
                            label="▶ Oynat",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=40, redraw=False),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Duraklat",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                        ),
                    ],
                )
            ]
        )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Animasyonu oynatmak için oynat ve duraklat butonlarına basınız.")


# -------------------------
# 6. Sonuçlar + AI Analizi + Sonifikasyon
# -------------------------
with col_res:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="small" style="font-weight:800;">GEÇİŞ OLASILIĞI (T)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">%{T*100:.3f}</div>', unsafe_allow_html=True)

    if E < V0:
        if T > 0.01:
            st.success("🌊 Kuantum Tünelleme")
        else:
            st.warning("🧱 Tünelleme Çok Zayıf")
    else:
        st.info("🚀 Klasik Geçiş (E ≥ V₀)")

    st.markdown(
        f"""
<div style="margin-top:12px; font-size:0.93rem; border-top:1px solid rgba(15,23,42,0.08); padding-top:10px;">
<b>Detaylı Veriler:</b><br>
• E / V₀ Oranı: <b>{E/V0:.2f}</b><br>
• Yansıma (R): <b>%{(1-T)*100:.2f}</b><br>
• Bariyer Faktörü (V₀·L): <b>{(V0*L):.2f}</b>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # AI Yorumlayıcı
    st.markdown("### 🧠 AI Analizi")
    if T < 0.0001:
        ai_text = "🚫 **Durum: Neredeyse Tam İzolasyon.** Bariyer çok yüksek/geniş; parçacığın karşıya geçme olasılığı çok çok düşük."
    elif T < 0.10:
        ai_text = "🌑 **Durum: Zayıf Tünelleme.** Çoğu parçacık yansır, çok azı bariyerin ötesine sızar."
    elif T < 0.90:
        ai_text = "🌓 **Durum: Kısmi Geçiş.** Enerji bariyere yakın; tünelleme ve geçiş birlikte görülür (STM mantığına benzer)."
    else:
        ai_text = "🌕 **Durum: Serbest Geçişe Yakın.** Enerji bariyere göre yeterli; klasik davranış baskın."

    st.markdown(f'<div class="ai-box">{ai_text}</div>', unsafe_allow_html=True)

    st.write("")

    # Sonifikasyon
    st.markdown("### 🎵 Kuantum Sesi")
    wav_bytes = generate_quantum_sound(T, E)
    st.audio(wav_bytes, format="audio/wav")


# -------------------------
# 7. Alt Bilgi
# -------------------------
st.markdown("---")
st.markdown(
    """
<div style="text-align:center; color:#94a3b8; font-size:0.86rem; padding:6px 0;">
    TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması<br>
  
</div>
""",
    unsafe_allow_html=True,
)
tüm kodu verme
