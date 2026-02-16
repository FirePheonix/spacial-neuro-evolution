"""
OpenStreetMap Data Loader for Spatiotemporal Traffic Analysis

This module handles loading and preprocessing road network data from OpenStreetMap
for use in geographically weighted traffic modeling.
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle


class OSMDataLoader:
    """Load and process OpenStreetMap data for traffic simulation."""
    
    def __init__(self, cache_dir: str = "data/osm_cache"):
        """
        Initialize the OSM data loader.
        
        Parameters
        ----------
        cache_dir : str
            Directory to cache downloaded OSM data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure OSMnx
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
    def load_city_network(
        self,
        place_name: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        network_type: str = 'drive',
        simplify: bool = True
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Load road network from OpenStreetMap.
        
        Parameters
        ----------
        place_name : str, optional
            Name of city/place to download (e.g., "San Francisco, California, USA")
        bbox : tuple, optional
            Bounding box (north, south, east, west) for the area
        network_type : str
            Type of street network ('drive', 'walk', 'bike', 'all')
        simplify : bool
            Whether to simplify the network topology
            
        Returns
        -------
        nodes : GeoDataFrame
            Network nodes (intersections)
        edges : GeoDataFrame
            Network edges (road segments)
        """
        cache_key = f"{place_name}_{bbox}_{network_type}".replace(" ", "_").replace(",", "")
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            print(f"Loading cached network from {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['nodes'], data['edges']
        
        # Download from OSM
        print(f"Downloading network from OpenStreetMap...")
        if place_name:
            G = ox.graph_from_place(place_name, network_type=network_type, simplify=simplify)
        elif bbox:
            G = ox.graph_from_bbox(
                bbox[0], bbox[1], bbox[2], bbox[3],
                network_type=network_type,
                simplify=simplify
            )
        else:
            raise ValueError("Either place_name or bbox must be provided")
        
        # Convert to GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(G)
        
        # Add useful features
        edges = self._enrich_edges(edges)
        nodes = self._enrich_nodes(nodes, edges)
        
        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump({'nodes': nodes, 'edges': edges}, f)
        
        print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
        return nodes, edges
    
    def _enrich_edges(self, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add additional features to edge data."""
        # Calculate edge length if not present
        if 'length' not in edges.columns:
            edges['length'] = edges.geometry.length
        
        # Classify road types
        edges['road_class'] = edges.apply(self._classify_road, axis=1)
        
        # Extract speed limits (if available)
        if 'maxspeed' in edges.columns:
            edges['speed_limit'] = edges['maxspeed'].apply(self._parse_speed)
        else:
            # Estimate based on road type
            edges['speed_limit'] = edges['road_class'].map({
                'highway': 65,
                'primary': 45,
                'secondary': 35,
                'residential': 25,
                'service': 15
            })
        
        # Number of lanes
        if 'lanes' in edges.columns:
            edges['n_lanes'] = pd.to_numeric(edges['lanes'], errors='coerce').fillna(2)
        else:
            edges['n_lanes'] = 2
        
        # One-way streets
        if 'oneway' in edges.columns:
            edges['is_oneway'] = edges['oneway'].astype(bool)
        else:
            edges['is_oneway'] = False
        
        return edges
    
    def _enrich_nodes(
        self,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Add additional features to node data."""
        # Calculate node degree (number of connected roads)
        node_degrees = edges.groupby('u').size() + edges.groupby('v').size()
        nodes['degree'] = nodes.index.map(lambda x: node_degrees.get(x, 0))
        
        # Classify intersection types
        nodes['intersection_type'] = nodes['degree'].apply(self._classify_intersection)
        
        return nodes
    
    @staticmethod
    def _classify_road(row) -> str:
        """Classify road into major categories."""
        highway_type = row.get('highway', 'residential')
        
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        
        if highway_type in ['motorway', 'motorway_link', 'trunk', 'trunk_link']:
            return 'highway'
        elif highway_type in ['primary', 'primary_link']:
            return 'primary'
        elif highway_type in ['secondary', 'secondary_link', 'tertiary', 'tertiary_link']:
            return 'secondary'
        elif highway_type in ['residential', 'living_street']:
            return 'residential'
        else:
            return 'service'
    
    @staticmethod
    def _parse_speed(speed_str) -> float:
        """Parse speed limit from OSM format."""
        if pd.isna(speed_str):
            return 25.0
        
        if isinstance(speed_str, list):
            speed_str = speed_str[0]
        
        try:
            # Remove 'mph' or 'km/h' and convert
            speed = float(str(speed_str).split()[0])
            return speed
        except:
            return 25.0
    
    @staticmethod
    def _classify_intersection(degree: int) -> str:
        """Classify intersection by the number of connected roads."""
        if degree <= 1:
            return 'dead_end'
        elif degree == 2:
            return 'continuation'
        elif degree == 3:
            return 't_junction'
        elif degree == 4:
            return 'crossroads'
        else:
            return 'complex'
    
    def load_pois(
        self,
        place_name: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        tags: Optional[Dict[str, List[str]]] = None
    ) -> gpd.GeoDataFrame:
        """
        Load Points of Interest (POIs) from OpenStreetMap.
        
        Parameters
        ----------
        place_name : str, optional
            Name of city/place
        bbox : tuple, optional
            Bounding box (north, south, east, west)
        tags : dict, optional
            OSM tags to query (e.g., {'amenity': ['restaurant', 'cafe']})
            
        Returns
        -------
        pois : GeoDataFrame
            Points of interest
        """
        if tags is None:
            # Default: load common POIs
            tags = {
                'amenity': ['school', 'hospital', 'restaurant', 'cafe', 'bar'],
                'shop': True,
                'office': True,
                'leisure': True
            }
        
        print("Downloading POIs from OpenStreetMap...")
        if place_name:
            pois = ox.features_from_place(place_name, tags=tags)
        elif bbox:
            pois = ox.features_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], tags=tags)
        else:
            raise ValueError("Either place_name or bbox must be provided")
        
        # Convert to point geometries (centroids for polygons)
        pois['geometry'] = pois.geometry.centroid
        
        return pois
    
    def compute_poi_density(
        self,
        locations: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        radius: float = 500.0
    ) -> np.ndarray:
        """
        Compute POI density around each location.
        
        Parameters
        ----------
        locations : GeoDataFrame
            Locations to compute density for
        pois : GeoDataFrame
            Points of interest
        radius : float
            Search radius in meters
            
        Returns
        -------
        density : ndarray
            POI count within radius for each location
        """
        densities = []
        
        for idx, loc in locations.iterrows():
            # Count POIs within radius
            distances = pois.geometry.distance(loc.geometry)
            count = (distances <= radius).sum()
            densities.append(count)
        
        return np.array(densities)


def create_sample_network(n_nodes: int = 50, seed: int = 42) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Create a simple synthetic road network for testing.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    nodes : GeoDataFrame
        Synthetic nodes
    edges : GeoDataFrame
        Synthetic edges
    """
    np.random.seed(seed)
    from shapely.geometry import Point, LineString
    
    # Create random nodes  
    xx = np.random.uniform(0, 1, n_nodes)
    yy = np.random.uniform(0, 1, n_nodes)
    
    # Create nodes GeoDataFrame
    nodes = gpd.GeoDataFrame(
        {
            'x': xx,
            'y': yy,
            'node_id': range(n_nodes)
        },
        geometry=[Point(x, y) for x, y in zip(xx, yy)],
        crs='EPSG:4326'
    )
    
    # Create edges by connecting nearby nodes
    edges_list = []
    
    for i in range(n_nodes):
        # Find nearby nodes
        distances = np.sqrt((xx - xx[i])**2 + (yy - yy[i])**2)
        nearby = np.where((distances > 0) & (distances < 0.15))[0]
        
        for j in nearby[:3]:  # Connect to at most 3 nearby nodes
            if i < j:  # Avoid duplicates
                edges_list.append({
                    'u': i,
                    'v': j,
                    'geometry': LineString([Point(xx[i], yy[i]), Point(xx[j], yy[j])]),
                    'length': distances[j] * 100000,  # Convert to meters
                    'road_class': np.random.choice(['highway', 'primary', 'residential'], p=[0.2, 0.3, 0.5])
                })
    
    edges = gpd.GeoDataFrame(edges_list, crs='EPSG:4326')
    
    # Add road classifications
    edges['speed_limit'] = edges['road_class'].map({
        'highway': 65,
        'primary': 45,
        'residential': 25
    })
    edges['n_lanes'] = edges['road_class'].map({
        'highway': 4,
        'primary': 2,
        'residential': 2
    })
    
    nodes['degree'] = 0
    for idx, edge in edges.iterrows():
        nodes.loc[edge['u'], 'degree'] += 1
        nodes.loc[edge['v'], 'degree'] += 1
    
    print(f"Created synthetic network with {len(nodes)} nodes and {len(edges)} edges")
    return nodes, edges


if __name__ == "__main__":
    # Example usage
    loader = OSMDataLoader()
    
    # Load a small area for testing
    # Using Berkeley, CA as an example
    nodes, edges = loader.load_city_network(
        place_name="Berkeley, California, USA",
        network_type='drive'
    )
    
    print(f"\nNetwork statistics:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"\nRoad class distribution:")
    print(edges['road_class'].value_counts())
    print(f"\nIntersection type distribution:")
    print(nodes['intersection_type'].value_counts())
