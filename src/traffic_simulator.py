"""
Spatiotemporal Traffic Pattern Simulator

This module generates realistic traffic patterns with spatial and temporal heterogeneity
for training geographically weighted models.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, Dict, Callable
from scipy.spatial import distance_matrix
from shapely.geometry import Point


class SpatiotemporalTrafficSimulator:
    """
    Simulate traffic patterns with spatial and temporal variation.
    
    This simulator creates realistic traffic data that exhibits:
    - Spatial heterogeneity (different patterns in different locations)
    - Temporal patterns (rush hour, off-peak periods)
    - Interaction effects (e.g., rush hour impact varies by location)
    """
    
    def __init__(
        self,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        seed: int = 42
    ):
        """
        Initialize the traffic simulator.
        
        Parameters
        ----------
        nodes : GeoDataFrame
            Road network nodes (intersections)
        edges : GeoDataFrame
            Road network edges (road segments)
        seed : int
            Random seed for reproducibility
        """
        self.nodes = nodes
        self.edges = edges
        self.rng = np.random.RandomState(seed)
        
        # Compute spatial zones based on clustering
        self._create_spatial_zones()
        
    def _create_spatial_zones(self, n_zones: int = 5):
        """
        Create spatial zones with different traffic characteristics.
        
        Zones represent different urban contexts (downtown, suburban, etc.)
        """
        from sklearn.cluster import KMeans
        
        # Get edge centroids
        centroids = np.array([
            [geom.centroid.x, geom.centroid.y]
            for geom in self.edges.geometry
        ])
        
        # Cluster into zones
        kmeans = KMeans(n_clusters=n_zones, random_state=42)
        self.edges['zone'] = kmeans.fit_predict(centroids)
        
        # Assign zone characteristics
        self.zone_characteristics = {
            0: {'type': 'downtown', 'base_congestion': 0.8, 'rush_sensitivity': 0.9},
            1: {'type': 'commercial', 'base_congestion': 0.6, 'rush_sensitivity': 0.7},
            2: {'type': 'residential', 'base_congestion': 0.3, 'rush_sensitivity': 0.5},
            3: {'type': 'suburban', 'base_congestion': 0.2, 'rush_sensitivity': 0.3},
            4: {'type': 'industrial', 'base_congestion': 0.4, 'rush_sensitivity': 0.4}
        }
        
        # Map to edges
        for zone_id, chars in self.zone_characteristics.items():
            mask = self.edges['zone'] == zone_id
            self.edges.loc[mask, 'zone_type'] = chars['type']
            self.edges.loc[mask, 'base_congestion'] = chars['base_congestion']
            self.edges.loc[mask, 'rush_sensitivity'] = chars['rush_sensitivity']
    
    def generate_traffic_patterns(
        self,
        n_samples: int = 5000,
        time_range: Tuple[float, float] = (0, 24),
        include_weather: bool = False,
        include_events: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Generate spatiotemporal traffic patterns.
        
        Parameters
        ----------
        n_samples : int
            Number of traffic observations to generate
        time_range : tuple
            Time range in hours (min, max)
        include_weather : bool
            Whether to include weather effects
        include_events : bool
            Whether to include special events
            
        Returns
        -------
        traffic_data : GeoDataFrame
            Generated traffic patterns with features and targets
        """
        # Sample edges
        edge_ids = self.rng.choice(len(self.edges), size=n_samples, replace=True)
        sampled_edges = self.edges.iloc[edge_ids].copy()
        
        # Sample times
        hours = self.rng.uniform(time_range[0], time_range[1], size=n_samples)
        
        # Create feature matrix
        data = {
            'hour': hours,
            'day_of_week': self.rng.randint(0, 7, size=n_samples),
            'road_class': sampled_edges['road_class'].values,
            'speed_limit': sampled_edges['speed_limit'].values,
            'n_lanes': sampled_edges['n_lanes'].values,
            'zone': sampled_edges['zone'].values,
            'zone_type': sampled_edges['zone_type'].values,
            'base_congestion': sampled_edges['base_congestion'].values,
            'rush_sensitivity': sampled_edges['rush_sensitivity'].values,
        }
        
        # Get geometry
        centroids = [geom.centroid for geom in sampled_edges.geometry]
        data['x'] = [p.x for p in centroids]
        data['y'] = [p.y for p in centroids]
        
        # Create GeoDataFrame
        traffic_gdf = gpd.GeoDataFrame(
            data,
            geometry=centroids,
            crs=self.edges.crs
        )
        
        # Generate targets (speed, congestion level, optimal maneuver)
        traffic_gdf = self._generate_targets(traffic_gdf)
        
        # Add weather effects if requested
        if include_weather:
            traffic_gdf = self._add_weather_effects(traffic_gdf)
        
        # Add event effects if requested
        if include_events:
            traffic_gdf = self._add_event_effects(traffic_gdf)
        
        return traffic_gdf
    
    def _generate_targets(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Generate target variables with spatial and temporal patterns.
        
        The key insight: relationships between features and targets vary by location!
        """
        n = len(gdf)
        
        # Initialize arrays
        optimal_speed = np.zeros(n)
        congestion_level = np.zeros(n)
        
        for i in range(n):
            hour = gdf.iloc[i]['hour']
            speed_limit = gdf.iloc[i]['speed_limit']
            n_lanes = gdf.iloc[i]['n_lanes']
            zone_type = gdf.iloc[i]['zone_type']
            base_cong = gdf.iloc[i]['base_congestion']
            rush_sens = gdf.iloc[i]['rush_sensitivity']
            x, y = gdf.iloc[i]['x'], gdf.iloc[i]['y']
            
            # Temporal pattern: rush hour effect
            rush_hour_factor = self._rush_hour_effect(hour)
            
            # Spatial pattern: location affects how rush hour impacts traffic
            # This is the key non-stationarity!
            location_modifier = self._spatial_pattern(x, y)
            
            # Congestion level (0 to 1)
            congestion = base_cong + rush_sens * rush_hour_factor
            congestion = congestion * (1 + 0.3 * location_modifier)
            congestion = np.clip(congestion, 0, 1)
            
            # Optimal speed depends on congestion and spatial location
            # Different relationships in different areas!
            speed_reduction = congestion * (0.5 + 0.3 * location_modifier)
            optimal = speed_limit * (1 - speed_reduction)
            
            # Lane effect varies by location (downtown: lanes matter more)
            if zone_type == 'downtown':
                optimal *= (1 + 0.1 * (n_lanes - 2))
            elif zone_type == 'residential':
                optimal *= (1 - 0.05 * (n_lanes - 2))  # Negative effect!
            
            # Add noise
            optimal += self.rng.normal(0, 3)
            congestion += self.rng.normal(0, 0.05)
            
            optimal_speed[i] = np.clip(optimal, 5, speed_limit)
            congestion_level[i] = np.clip(congestion, 0, 1)
        
        gdf['optimal_speed'] = optimal_speed
        gdf['congestion_level'] = congestion_level
        
        # Generate categorical target: optimal maneuver
        gdf['optimal_maneuver'] = gdf.apply(self._determine_maneuver, axis=1)
        
        # Generate continuous target for acceleration
        gdf['optimal_acceleration'] = self._generate_acceleration(gdf)
        
        return gdf
    
    @staticmethod
    def _rush_hour_effect(hour: float) -> float:
        """
        Compute rush hour effect (0 to 1).
        
        Higher values during morning (7-9 AM) and evening (5-7 PM) rush hours.
        """
        # Morning rush hour (7-9 AM)
        morning = np.exp(-0.5 * ((hour - 8) / 0.7) ** 2)
        
        # Evening rush hour (5-7 PM)
        evening = np.exp(-0.5 * ((hour - 18) / 0.7) ** 2)
        
        return max(morning, evening)
    
    @staticmethod
    def _spatial_pattern(x: float, y: float) -> float:
        """
        Generate spatial pattern (-1 to 1).
        
        This creates smooth spatial variation in traffic patterns.
        """
        # Combination of sinusoidal patterns
        pattern = (
            0.3 * np.sin(2 * np.pi * x) +
            0.3 * np.cos(2 * np.pi * y) +
            0.2 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y) +
            0.2 * np.sin(np.pi * (x + y))
        )
        return pattern
    
    @staticmethod
    def _determine_maneuver(row) -> str:
        """
        Determine optimal maneuver based on conditions.
        
        Returns: 'accelerate', 'maintain', 'decelerate', 'stop'
        """
        congestion = row['congestion_level']
        speed_ratio = row['optimal_speed'] / row['speed_limit']
        
        if congestion > 0.8 or speed_ratio < 0.3:
            return 'stop'
        elif congestion > 0.6 or speed_ratio < 0.5:
            return 'decelerate'
        elif speed_ratio > 0.8:
            return 'maintain'
        else:
            return 'accelerate'
    
    def _generate_acceleration(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Generate optimal acceleration values.
        
        Positive for speeding up, negative for slowing down.
        """
        acceleration = np.zeros(len(gdf))
        
        for i in range(len(gdf)):
            congestion = gdf.iloc[i]['congestion_level']
            speed_limit = gdf.iloc[i]['speed_limit']
            optimal_speed = gdf.iloc[i]['optimal_speed']
            
            # Higher congestion -> more braking
            if congestion > 0.7:
                acceleration[i] = -2.0 - congestion * 2
            elif congestion > 0.4:
                acceleration[i] = -1.0
            elif optimal_speed < speed_limit * 0.8:
                acceleration[i] = 1.5
            else:
                acceleration[i] = 0.5
            
            # Add noise
            acceleration[i] += self.rng.normal(0, 0.3)
        
        return acceleration
    
    def _add_weather_effects(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add weather effects to traffic patterns."""
        # Simulate weather conditions
        weather_conditions = self.rng.choice(
            ['clear', 'rain', 'fog', 'snow'],
            size=len(gdf),
            p=[0.7, 0.2, 0.05, 0.05]
        )
        gdf['weather'] = weather_conditions
        
        # Adjust speeds based on weather
        weather_impact = {
            'clear': 1.0,
            'rain': 0.85,
            'fog': 0.70,
            'snow': 0.60
        }
        
        for weather, factor in weather_impact.items():
            mask = gdf['weather'] == weather
            gdf.loc[mask, 'optimal_speed'] *= factor
        
        return gdf
    
    def _add_event_effects(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add special event effects (concerts, sports, etc.)."""
        # 5% of observations have special events
        has_event = self.rng.random(len(gdf)) < 0.05
        gdf['has_event'] = has_event
        
        # Events increase congestion in that area
        gdf.loc[has_event, 'congestion_level'] = np.minimum(
            gdf.loc[has_event, 'congestion_level'] + 0.3,
            1.0
        )
        gdf.loc[has_event, 'optimal_speed'] *= 0.7
        
        return gdf
    
    def create_spatiotemporal_weights(
        self,
        locations: gpd.GeoDataFrame,
        times: np.ndarray,
        spatial_bandwidth: float = 0.1,
        temporal_bandwidth: float = 2.0,
        kernel: str = 'gaussian'
    ) -> np.ndarray:
        """
        Create spatiotemporal weight matrix using product-of-kernels.
        
        Parameters
        ----------
        locations : GeoDataFrame
            Spatial locations
        times : ndarray
            Temporal coordinates (hours)
        spatial_bandwidth : float
            Bandwidth for spatial kernel
        temporal_bandwidth : float
            Bandwidth for temporal kernel (hours)
        kernel : str
            Kernel type ('gaussian', 'bisquare', 'exponential')
            
        Returns
        -------
        weights : ndarray
            Spatiotemporal weight matrix (n x n)
        """
        n = len(locations)
        
        # Spatial distances
        coords = np.array([[p.x, p.y] for p in locations.geometry])
        spatial_dist = distance_matrix(coords, coords)
        
        # Temporal distances
        temporal_dist = np.abs(times[:, np.newaxis] - times[np.newaxis, :])
        
        # Apply kernels
        if kernel == 'gaussian':
            spatial_weights = np.exp(-0.5 * (spatial_dist / spatial_bandwidth) ** 2)
            temporal_weights = np.exp(-0.5 * (temporal_dist / temporal_bandwidth) ** 2)
        elif kernel == 'bisquare':
            spatial_weights = np.maximum(0, 1 - (spatial_dist / spatial_bandwidth) ** 2) ** 2
            temporal_weights = np.maximum(0, 1 - (temporal_dist / temporal_bandwidth) ** 2) ** 2
        elif kernel == 'exponential':
            spatial_weights = np.exp(-spatial_dist / spatial_bandwidth)
            temporal_weights = np.exp(-temporal_dist / temporal_bandwidth)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        # Product of kernels
        weights = spatial_weights * temporal_weights
        
        # Normalize rows
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = weights / (row_sums + 1e-10)
        
        return weights


def visualize_traffic_patterns(
    traffic_data: gpd.GeoDataFrame,
    variable: str = 'optimal_speed',
    time_slice: Optional[float] = None
):
    """
    Visualize spatial traffic patterns.
    
    Parameters
    ----------
    traffic_data : GeoDataFrame
        Traffic data to visualize
    variable : str
        Variable to plot
    time_slice : float, optional
        If provided, filter to specific hour
    """
    import matplotlib.pyplot as plt
    
    data = traffic_data.copy()
    
    if time_slice is not None:
        # Filter to specific time
        data = data[np.abs(data['hour'] - time_slice) < 0.5]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot as scatter with color
    scatter = ax.scatter(
        data.geometry.x,
        data.geometry.y,
        c=data[variable],
        s=50,
        alpha=0.6,
        cmap='RdYlGn_r' if 'congestion' in variable else 'RdYlGn'
    )
    
    plt.colorbar(scatter, ax=ax, label=variable)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    title = f'Spatial Pattern: {variable}'
    if time_slice is not None:
        title += f' at hour {time_slice:.1f}'
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage with synthetic network
    from data_loader import create_sample_network
    
    print("Creating sample network...")
    nodes, edges = create_sample_network(n_nodes=100)
    
    print("\nInitializing traffic simulator...")
    simulator = SpatiotemporalTrafficSimulator(nodes, edges)
    
    print("\nGenerating traffic patterns...")
    traffic_data = simulator.generate_traffic_patterns(n_samples=2000)
    
    print(f"\nGenerated {len(traffic_data)} traffic observations")
    print("\nSample data:")
    print(traffic_data[['hour', 'zone_type', 'optimal_speed', 'congestion_level', 'optimal_maneuver']].head(10))
    
    print("\nStatistics by zone type:")
    print(traffic_data.groupby('zone_type')['optimal_speed'].describe())
    
    print("\nDemonstrating spatial non-stationarity:")
    print("Speed reduction during rush hour varies by zone:")
    rush_hour = traffic_data[traffic_data['hour'].between(7, 9)]
    off_peak = traffic_data[traffic_data['hour'].between(13, 15)]
    
    for zone in traffic_data['zone_type'].unique():
        rush_speed = rush_hour[rush_hour['zone_type'] == zone]['optimal_speed'].mean()
        off_speed = off_peak[off_peak['zone_type'] == zone]['optimal_speed'].mean()
        print(f"  {zone}: Rush={rush_speed:.1f} mph, Off-peak={off_speed:.1f} mph, "
              f"Reduction={((off_speed - rush_speed) / off_speed * 100):.1f}%")
