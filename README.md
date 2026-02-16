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

The core of this project relies on **Geographically Weighted Regression (GWR)**, implemented via the **`gwlearn`** library (a Scikit-Learn compatible wrapper for `libpysal`).

1.  **Data Generation**:
    The system procedurally generates a synthetic city with non-stationary traffic rules. For example:
    - **Downtown**: High base congestion, severe evening rush hour.
    - **Suburbs**: Low congestion, moderate morning rush hour.
    - **Industrial**: Constant heavy truck traffic.

2.  **Model Training (Global vs. Spatial)**:
    - **Global Model (`sklearn.LinearRegression`)**:
      Trains a single linear equation on the entire dataset. It averages out the differences between zones, leading to high error rates in complex areas.
      $$ y = \beta_0 + \beta_1 x_1 + ... + \epsilon $$

    - **Spatial Model (`gwlearn.GWLinearRegression`)**:
      This is where **`gwlearn`** shines. It fits a separate regression model for *every single point in space*. When predicting traffic at Location $u$, it weights training samples based on their distance to $u$ using a kernel function (e.g., Bisquare or Gaussian).
      $$ y(u) = \beta_0(u) + \beta_1(u) x_1 + ... + \epsilon $$
      
      This allows the model to "learn" that the correlation between *Time of Day* and *Traffic Speed* is negative in the city center (rush hour slows you down) but might be positive in residential areas during the day.

3.  **Visualizing the "Brain"**:
    The web app (`app.py`) serves these predictions to a JavaScript frontend.
    - **Red cars** follow the Global Model's predictions (often speeding in congested zones).
    - **Blue cars** follow the **`gwlearn`** Spatial Model's predictions (slowing down appropriately in traffic).
    - Heatmaps show the **local $R^2$** values, proving that the spatial model accurately captures the unique dynamics of each neighborhood.
