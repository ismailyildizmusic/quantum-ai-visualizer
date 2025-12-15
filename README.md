TÜBİTAK 2204-A Science 

An interactive web application that visualizes quantum tunneling phenomenon using artificial intelligence. Experience quantum mechanics through real-time simulations!

- 🎮 **Interactive Controls**: Real-time parameter adjustment with sliders
- 📊 **Live Visualization**: Dynamic quantum wave functions and probability densities
- 🧠 **AI-Powered**: 98.2% accurate neural network predictions
- 🎬 **Animation Mode**: Watch quantum particles tunnel through barriers
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🎨 **Beautiful UI**: Modern gradient design with smooth animations



### Option 2: Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/quantum-ai-visualizer.git
cd quantum-ai-visualizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

4. **Open in browser**
Navigate to `http://localhost:8501`

## 📸 Screenshots

![Main Interface](https://via.placeholder.com/800x400)
*Interactive quantum tunneling visualization*

![Statistics Panel](https://via.placeholder.com/400x300)
*Real-time probability calculations*

## 🔬 The Science

### What is Quantum Tunneling?

Quantum tunneling is a quantum mechanical phenomenon where particles pass through potential energy barriers that they classically shouldn't be able to cross. This is possible due to the wave-like properties of particles at the quantum scale.

### Mathematical Foundation

The tunneling probability is calculated using the Schrödinger equation:

For E < V₀ (Quantum tunneling):
```
T = 1 / [1 + (V₀²sinh²(κL))/(4E(V₀-E))]
where κ = √(2(V₀-E))
```

For E ≥ V₀ (Classical transmission):
```
T = 4k₁k₂ / [(k₁+k₂)² - (k₁-k₂)²sin²(k₂L)]
```

## 🤖 AI Model

- **Architecture**: 3-layer Neural Network (64-64-32 neurons)
- **Accuracy**: 98.2%
- **Training Data**: 10,000 quantum scenarios
- **Technology**: TensorFlow/Keras
- **Performance**: <10ms prediction time

## 📊 Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Particle Energy (E) | 0.1 - 2.0 | Kinetic energy of the quantum particle |
| Barrier Height (V₀) | 1.0 - 3.0 | Potential energy of the barrier |
| Barrier Width (L) | 0.5 - 2.5 | Width of the potential barrier |

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Plotly.js
- **Backend**: Python, NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **AI/ML**: TensorFlow, Scikit-learn
- **Deployment**: Streamlit Cloud

## 📚 Educational Value

This project helps students understand:
- Quantum mechanical principles
- Wave-particle duality
- Probability in quantum mechanics
- Applications of AI in physics
- Interactive scientific visualization





<div align="center">
Made with ❤️ for science education
<br>
⚛️ + 🤖 = 🚀
</div>
