"""
Quantum AI Visualizer - Streamlit Web Application
TÜBİTAK 2204-A Project

Bilimsel not:
- Hesaplamalarda doğal birimler kullanılmıştır: ħ = 1, m = 1 (boyutsuz/normalize).
- Re(ψ) faz ile değişir; |ψ|² durağan durum için fazdan bağımsızdır.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# Sayfa ayarları
# -------------------------
st.set_page_config(
    page_title="Quantum AI Visualizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern/sade stil
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #f6f7fb;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.card {
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #0f172a;
}
.subtle {
    color: rgba(15, 23, 42, 0.65);
    font-size: 0.95rem;
}
hr {
    border: none;
    border-top: 1px solid rgba(15, 23, 42, 0.10);
    margin: 10px 0 14px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# Üst başlık
st.markdown(
    """
<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
      <div style="font-size: 2.05rem; font-weight: 900; color:#0f172a; line-height:1.1;">
        ⚛️ Quantum AI Visualizer
      </div>
      <div class="subtle" style="margin-top:6px;">
        Kuantum tünelleme • Dikdörtgen bariyer • Etkileşimli görselleştirme
      </div>
      <div style="margin-top:10px;">
        <span class="badge">🏆 TÜBİTAK 2204-A Ortaokul Projesi</span>
      </div>
    </div>
    <div class="subtle" style="text-align:right; min-width:220px;">
      Doğal birimler: ħ = 1, m = 1<br/>
      Eğitim amaçlı bilimsel simülasyon
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# -------------------------
# Fizik: Bariyer saçılması (1D) - sınır koşulları ile çözüm
# -------------------------
def solve_scattering_coeffs(E: float, V0: float, L: float):
    """
    Bölge I (x<0):     ψ = e^{ikx} + r e^{-ikx}
    Bölge II (0<x<L):  ψ = A e^{iqx} + B e^{-iqx}   (q gerçek ya da imajiner)
    Bölge III (x>L):   ψ = t e^{ikx}

    Doğal birimler: ħ=1, m=1
    k = sqrt(2E)
    E<V0 ise q = iκ, κ = sqrt(2(V0-E))
    """
    E = max(E, 1e-6)
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

    # Sol ve sağ tarafta potansiyel aynı (0) olduğundan T = |t|^2
    T = float(np.clip(np.abs(t) ** 2, 0.0, 1.0))

    return r, A, B, t, T, k, q


def psi_x(x: np.ndarray, E: float, V0: float, L: float, phase: float):
    """
    ψ(x) hesaplar. Global faz exp(-i*phase) ile Re(ψ) hareket eder.
    |ψ|^2 fazdan bağımsızdır (durağan durum).
    """
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


def build_figure(E: float, V0: float, L: float, animate: bool, n_frames: int):
    x = np.linspace(-3.0, L + 3.0, 900)

    # Potansiyel
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= L)] = V0

    # T
    _, _, _, _, T, _, _ = solve_scattering_coeffs(E, V0, L)

    # Sabit: |ψ|^2 (fazdan bağımsız)
    psi0 = psi_x(x, E, V0, L, phase=0.0)
    prob = np.abs(psi0) ** 2

    # Re(ψ) başlangıç
    re0 = np.real(psi0)
    re_norm = re0 / (np.max(np.abs(re0)) + 1e-9)

    # Re(ψ) çizimini E seviyesine yakın ölçekleyelim (okunabilirlik için)
    scale = 0.28 * max(1.0, V0)
    re_y0 = E + scale * re_norm

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=(
            "Üst: Potansiyel ve Enerji • Re(ψ) (animasyon)",
            f"Alt: Olasılık Yoğunluğu |ψ|² • Tünelleme Olasılığı T = {T:.4f}",
        ),
    )

    # Üst: bariyer
    fig.add_trace(
        go.Scatter(
            x=x,
            y=V,
            name="V(x)",
            fill="tozeroy",
            opacity=0.35,
            line=dict(width=2),
        ),
        row=1, col=1
    )

    # Üst: enerji
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[E] * len(x),
            name="E",
            line=dict(width=2, dash="dash"),
        ),
        row=1, col=1
    )

    # Üst: Re(ψ)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=re_y0,
            name="Re(ψ) (ölçekli)",
            line=dict(width=2),
            opacity=0.95,
        ),
        row=1, col=1
    )

    # Alt: |ψ|^2
    fig.add_trace(
        go.Scatter(
            x=x,
            y=prob,
            name="|ψ|²",
            fill="tozeroy",
            opacity=0.35,
            line=dict(width=2),
        ),
        row=2, col=1
    )

    # Bariyer sınırları
    for r in [1, 2]:
        fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="gray", row=r, col=1)
        fig.add_vline(x=L, line_dash="dot", line_width=1, line_color="gray", row=r, col=1)

    fig.update_xaxes(title_text="Konum x", row=2, col=1)
    fig.update_yaxes(title_text="Enerji / Potansiyel", row=1, col=1)
    fig.update_yaxes(title_text="Olasılık Yoğunluğu", row=2, col=1)

    fig.update_layout(
        height=780,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=28, r=28, t=70, b=28),
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Animasyon: Plotly frames + play/pause (Streamlit rerun yok!)
    if animate:
        phases = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)

        frames = []
        for ph in phases:
            psi = psi_x(x, E, V0, L, phase=float(ph))
            re = np.real(psi)
            re_norm = re / (np.max(np.abs(re)) + 1e-9)
            re_y = E + scale * re_norm

            # 3. trace (index=2) Re(ψ) trace'idir
            frames.append(
                go.Frame(
                    data=[go.Scatter(x=x, y=re_y)],
                    name=f"{ph:.3f}",
                    traces=[2],
                )
            )

        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.02,
                    y=1.12,
                    buttons=[
                        dict(
                            label="▶ Oynat",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=35, redraw=False),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Duraklat",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                            ],
                        ),
                    ],
                )
            ]
        )

    return fig, T


# -------------------------
# Yan panel (Türkçe)
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ Kontrol Paneli")
    st.caption("Not: Doğal birimler (ħ=1, m=1) kullanılır. Değerler boyutsuzdur.")

    energy = st.slider("⚡ Parçacık Enerjisi (E)", 0.10, 2.00, 0.80, 0.01)
    barrier_height = st.slider("📈 Bariyer Yüksekliği (V₀)", 1.00, 3.00, 1.50, 0.01)
    barrier_width = st.slider("↔️ Bariyer Genişliği (L)", 0.50, 2.50, 1.00, 0.01)

    st.divider()

    animate = st.toggle("🎞️ Re(ψ) Animasyonu", value=True)
    n_frames = st.slider("Animasyon Akıcılığı (kare)", 20, 120, 60, 5)

    st.divider()

    st.markdown("### 🎭 Hazır Senaryolar")
    preset = st.radio(
        "Seç:",
        ["Özel", "Güçlü Tünelleme", "Zayıf Tünelleme", "Klasik Geçiş", "Kritik Nokta"],
        index=0,
    )

    if preset == "Güçlü Tünelleme":
        energy, barrier_height, barrier_width = 0.90, 1.00, 0.50
    elif preset == "Zayıf Tünelleme":
        energy, barrier_height, barrier_width = 0.25, 2.50, 2.00
    elif preset == "Klasik Geçiş":
        energy, barrier_height, barrier_width = 1.80, 1.20, 1.00
    elif preset == "Kritik Nokta":
        energy, barrier_height, barrier_width = 1.50, 1.50, 1.00


# -------------------------
# Ana görünüm
# -------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig, T = build_figure(
        E=float(energy),
        V0=float(barrier_height),
        L=float(barrier_width),
        animate=bool(animate),
        n_frames=int(n_frames),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    regime = "Klasik geçiş (E ≥ V₀)" if energy >= barrier_height else "Kuantum tünelleme (E < V₀)"
    st.markdown(
        f"""
<div class="card">
  <div style="font-size:1.05rem; font-weight:800; color:#0f172a;">📊 Sonuçlar</div>
  <hr/>
  <div class="subtle">Tünelleme Olasılığı (T)</div>
  <div style="font-size:2.6rem; font-weight:900; color:#0f172a; margin-top:4px;">{T:.4f}</div>
  <div class="subtle" style="margin-top:4px;">%{T*100:.2f}</div>
  <hr/>
  <div class="subtle">Durum</div>
  <div style="font-weight:800; color:#0f172a; margin-top:4px;">{regime}</div>
  <hr/>
  <div class="subtle">Hızlı oranlar</div>
  <div style="margin-top:6px;">E/V₀ = <b>{energy/barrier_height:.3f}</b></div>
  <div>Yansıma ≈ <b>{(1-T)*100:.1f}%</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander("✅ Bilimsel açıklama (kısa)"):
        st.markdown(
            """
- Bu uygulama, *1 boyutlu dikdörtgen bariyer* için Schrödinger denkleminin sınır koşullarını çözerek ψ(x) katsayılarını bulur.
- Görülen animasyon *Re(ψ)* bileşeninin faz ile değişimidir; *|ψ|²* durağan durumda fazdan bağımsızdır.
- Tünelleme olasılığı *T = |t|²* olarak hesaplanır (sol/sağ potansiyel aynı: 0).
"""
        )

st.write("")
st.caption("Not: Animasyon için grafiğin üstündeki ▶ Oynat / ⏸ Duraklat düğmelerini kullan. (Tübitak 2204 - A Kapsamında Hazırlanmıştır.)")
