<h1 align="center">Spatiotemporal AV Navigation</h1>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/gwlearn-Spatial_Model-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/GeoPandas-Spatial_Data-139C5A?style=for-the-badge&logo=geopandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c67c61bb-9f2d-448b-8268-89072f96c811" />
</p>


A neuroevolution-based traffic simulation and prediction engine where autonomous vehicles learn to navigate from one point to another without going off the road while adapting to spatial and temporal traffic dynamics. The vehicles evolve their driving behavior through neuroevolution for path optimization, while a spatially-aware intelligence layer powered by Geographically Weighted Regression (GWR) using gwlearn(https://github.com/pysal/gwlearn) enables them to dynamically adjust speed and movement based on zone type, time of day, and local traffic conditions. By comparing global models with spatially-aware models, the project demonstrates how location-specific intelligence significantly improves autonomous navigation in non-stationary urban environments.

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

<img width="1662" height="834" alt="image" src="https://github.com/user-attachments/assets/19d5152f-699f-4341-aabc-912006f67f97" />


1.  **Data Generation**:
    The system procedurally generates a synthetic city with non-stationary traffic rules. For example:
    - **Downtown**: High base congestion, severe evening rush hour.
    - **Suburbs**: Low congestion, moderate morning rush hour.
    - **Industrial**: Constant heavy truck traffic.

2. **Model Training (Global vs. Spatial)**

- **Global Model (`sklearn.LinearRegression`)**  
  Trains a single linear equation on the entire dataset. It averages out the differences between zones, leading to high error rates in complex areas.

  `y = beta_0 + beta_1 * x_1 + ... + epsilon`

- **Spatial Model (`gwlearn.GWLinearRegression`)**  
  This is where `gwlearn` shines. It fits a separate regression model for every single point in space. When predicting traffic at location `u`, it weights training samples based on their distance to `u` using a kernel function (e.g., Bisquare or Gaussian).

  `y(u) = beta_0(u) + beta_1(u) * x_1 + ... + epsilon`

      
  This allows the model to "learn" that the correlation between *Time of Day* and *Traffic Speed* is negative in the city center (rush hour slows you down) but might be positive in residential areas during the day.

3.  **Visualizing the "Brain"**:
    The web app (`app.py`) serves these predictions to a JavaScript frontend.
    - **Red cars** follow the Global Model's predictions (often speeding in congested zones).
    - **Blue cars** follow the **`gwlearn`** Spatial Model's predictions (slowing down appropriately in traffic).
    - Heatmaps show the **local $R^2$** values, proving that the spatial model accurately captures the unique dynamics of each neighborhood.
  
4.  **Neuro-Evolution (Two Layers of AI)**:
    This project features two distinct evolutionary systems:
    
    *   **Layer 1: Spatial Intelligence (The "Oracle")**  
        *Runs in Python (Backend)*  
        Uses a Genetic Algorithm to evolve the optimal **Spatial Bandwidth** for `gwlearn`. It learns *how* to analyze the city's traffic zones (e.g., "how far should I look to judge traffic density?").
        
    *   **Layer 2: Navigation Intelligence (The "Driver")**  
        *Runs in JavaScript (Frontend)*  
        Each car is controlled by a **Neural Network** that takes inputs from:
        1.  **LIDAR Sensors**: Raycasts detecting road borders and other cars.
        2.  **GPS Vector**: The angle to the target destination.
        
        Using **Genetic Algorithms**, the cars evolve over generations. They start as random crashers but eventually "learn" to:
        - Steer smoothly around curves.
        - Avoid collisions with other vehicles.
        - Navigate towards a target destination efficiently.
