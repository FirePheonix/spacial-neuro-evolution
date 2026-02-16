# Spatiotemporal AV Navigator

A traffic simulation and prediction engine that models how traffic speed varies across different zones of a city. It compares Global AI models against Spatially-Aware models to demonstrate the importance of location-specific intelligence in autonomous navigation.

## Installation

### Prerequisites
- Python 3.8+
- Git

### Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spatiotemporal-av-navigator.git
   cd spatiotemporal-av-navigator
   ```

2. Run the setup script (Windows):
   ```bash
   SETUP_AND_RUN.bat
   ```

   **Or manually:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

## Usage

### 1. Web Simulation (Interactive)
Launch the full interactive visualization server:
```bash
python app.py
```
Open your browser to `http://localhost:5000`. This provides:
- Real-time traffic simulation
- Interactive route planning
- Comparison dashboards between Global and Spatial models
- Heatmap visualizations

### 2. Examples
Run the standalone examples to see the data science behind the project.

**Example 1: Basic Traffic Modeling**
Generates a synthetic city and proves that spatial models outperform global averages.
```bash
python examples/01_basic_traffic_modeling.py
```

**Example 2: Evolving Spatial Intelligence**
Uses a Genetic Algorithm to evolve the optimal "spatial bandwidth" for a specific city layout.
```bash
python examples/02_evolving_spatial_intelligence.py
```

## About the Project

Traditional traffic prediction often treats an entire city as a single dataset (Global Model), ignoring local nuances. This project demonstrates that a **Spatially-Aware Model**—which weighs nearby data points more heavily than distant ones—can significantly reduce prediction error.

### Key Features
- **Procedural City Generation**: Creates realistic road networks with distinct zones (Downtown, Residential, Industrial).
- **Temporal Dynamics**: Simulates rush hour patterns that vary by zone (e.g., Downtown is congested at 5 PM, Suburbs at 8 AM).
- **Spatial vs. Global AI**: Benchmarks standard Linear Regression against Geographically Weighted Regression (GWR).
- **Neuro-Evolution**: Implements genetic algorithms to optimize spatial parameters automatically.

## How it Works

1.  **Data Generation**: The system generates thousands of traffic samples with localized rules.
2.  **Model Training**:
    *   **Global Model**: Learns one equation for the whole map ($y = mx + b$). Good generalist, but misses local details.
    *   **Spatial Model**: Learns local equations for every point on the map. It understands that "5 PM" means "slow" in Downtown but "fast" in Outskirts.
3.  **Visualization**: The results are rendered using a custom HTML5 canvas engine and Matplotlib dashboards.
