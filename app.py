"""
Spatiotemporal AV Navigator — Visual Simulation Server

Flask backend serving:
  - Spatial traffic predictions (Python)
  - Interactive Canvas simulation (HTML5/JS)
"""

import os, sys, json, math, random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# --- Spatial Model -----------------------------------------------------------------
from gwlearn.linear_model import GWLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# World generation helpers
# ---------------------------------------------------------------------------

random.seed(42)
np.random.seed(42)

ZONE_META = {
    "downtown":    {"color": "#e74c3c", "base_cong": 0.7, "rush_sens": 0.9, "speed": 25},
    "commercial":  {"color": "#e67e22", "base_cong": 0.5, "rush_sens": 0.7, "speed": 35},
    "residential": {"color": "#2ecc71", "base_cong": 0.2, "rush_sens": 0.3, "speed": 35},
    "suburban":    {"color": "#3498db", "base_cong": 0.1, "rush_sens": 0.2, "speed": 45},
    "industrial":  {"color": "#9b59b6", "base_cong": 0.4, "rush_sens": 0.5, "speed": 40},
}


# ---------------------------------------------------------------------------
# World generation / loading
# ---------------------------------------------------------------------------

WORLD_FILE = os.path.join("static", "world", "saves", "big.world")

def _load_big_world():
    """Load the actual 'big.world' JSON file used by the frontend."""
    if not os.path.exists(WORLD_FILE):
        print(f"WARNING: Coould not find {WORLD_FILE}, falling back to synthetic.")
        return _build_world() # Fallback
    
    print(f"Loading world from {WORLD_FILE}...")
    with open(WORLD_FILE, "r") as f:
        # The file content is "const world = World.load({...})"
        # We need to extract the JSON object.
        content = f.read()
        json_str = content[content.find("{"):content.rfind("}")+1]
        data = json.loads(json_str)
        
    raw_nodes = data["graph"]["points"]
    raw_edges = data["graph"]["segments"]
    
    # Convert to our format
    nodes = []
    ids_map = {} # id -> index
    
    xs = []
    ys = []
    
    for i, p in enumerate(raw_nodes):
        nodes.append({
            "id": p["id"],
            "x": float(p["x"]), 
            "y": float(p["y"])
        })
        ids_map[p["id"]] = i
        xs.append(p["x"])
        ys.append(p["y"])
        
    edges = []
    for seg in raw_edges:
        # Segments in World.js usually have p1, p2 objects or ids
        # In this big.world, they look like {p1: {id:...}, p2: {id:...}}
        try:
            u_id = seg["p1"]["id"]
            v_id = seg["p2"]["id"]
            if u_id in ids_map and v_id in ids_map:
                u_node = nodes[ids_map[u_id]]
                v_node = nodes[ids_map[v_id]]
                dist = math.hypot(u_node["x"] - v_node["x"], u_node["y"] - v_node["y"])
                edges.append({
                    "u": u_id, "v": v_id,
                    "length": dist,
                    "road_class": "primary" # simplistic for now
                })
        except: pass

    # Assign zones via clustering on ACTUAL coordinates
    from sklearn.cluster import KMeans
    coords = np.column_stack([xs, ys])
    n_zones = 5
    # Use more clusters to get a better spread if map is huge
    km = KMeans(n_clusters=n_zones, n_init=10, random_state=42)
    labels = km.fit_predict(coords)
    
    zone_names = list(ZONE_META.keys())
    # Sort clusters by X coordinate to make them somewhat deterministic/spatial
    # (e.g. West is one zone, East is another)
    centers = km.cluster_centers_
    sorted_indices = np.argsort(centers[:, 0]) # sort by x
    
    # Map cluster index -> sorted rank -> zone name
    rank_map = {old: new for new, old in enumerate(sorted_indices)}
    
    for i, n in enumerate(nodes):
        cluster_id = labels[i]
        rank = rank_map[cluster_id]
        n["zone"] = zone_names[rank % len(zone_names)]

    return nodes, edges

def _build_world(n_nodes=60):
    """Fallback synthetic generator."""
    # ... (same as before, simplified for brevity in this replace block if needed, but I'll keep logic)
    np.random.seed(42)
    xs = np.random.uniform(80, 1520, n_nodes)
    ys = np.random.uniform(80, 720, n_nodes)
    nodes = [{"id": i, "x": float(xs[i]), "y": float(ys[i])} for i in range(n_nodes)]
    edges = [] # ... dummy edges
    
    # assign zones
    from sklearn.cluster import KMeans
    coords = np.column_stack([xs, ys])
    labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(coords)
    zone_names = list(ZONE_META.keys())
    for i, n in enumerate(nodes):
        n["zone"] = zone_names[labels[i] % len(zone_names)]
    return nodes, edges

WORLD_NODES, WORLD_EDGES = _load_big_world()

# ---------------------------------------------------------------------------
# Spatial model training
# ---------------------------------------------------------------------------

def _train_models():
    """Generate spatial traffic data and train both Global and Spatial models."""
    nodes = WORLD_NODES
    if not nodes: return {}
    
    n_samples = 2000 # More samples for bigger world
    rows = []
    
    # Get Bounds
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    for _ in range(n_samples):
        node = random.choice(nodes)
        hour = random.uniform(0, 24)
        zone = node["zone"]
        meta = ZONE_META[zone]

        rush_factor = max(0, 1 - abs(hour - 8)/3) + max(0, 1 - abs(hour - 17.5)/3)
        # Normalize loc_mod to be reasonable regardless of map scale
        nx = (node["x"] - min_x) / (max_x - min_x + 1)
        ny = (node["y"] - min_y) / (max_y - min_y + 1)
        
        loc_mod = math.sin(nx * 10) * math.cos(ny * 10)

        congestion = meta["base_cong"] + meta["rush_sens"] * rush_factor * (1 + 0.3*loc_mod)
        congestion = max(0, min(1, congestion))
        optimal_speed = meta["speed"] * (1 - congestion * (0.5 + 0.2*loc_mod))
        optimal_speed = max(5, optimal_speed + np.random.normal(0, 2))

        rows.append({
            "x": node["x"], "y": node["y"],
            "hour": hour, "speed_limit": meta["speed"],
            "base_congestion": meta["base_cong"],
            "rush_sensitivity": meta["rush_sens"],
            "congestion": congestion,
            "optimal_speed": optimal_speed,
            "zone": zone,
        })

    df = pd.DataFrame(rows)
    geometry = gpd.GeoSeries([Point(r["x"], r["y"]) for r in rows])

    feature_cols = ["hour", "speed_limit", "base_congestion", "rush_sensitivity"]
    X_df = df[feature_cols].copy() 
    y_s  = pd.Series(df["optimal_speed"].values, name="optimal_speed")

    # --- Global model ---
    global_model = LinearRegression()
    global_model.fit(X_df.values, y_s.values)
    global_pred = global_model.predict(X_df.values)
    global_rmse = float(np.sqrt(mean_squared_error(y_s, global_pred)))
    global_r2   = float(r2_score(y_s, global_pred))

    # --- Spatial model ---
    from libpysal.graph import Graph as PGraph
    # Adaptive bandwidth might be better for clustered data
    gw_graph = PGraph.build_kernel(
        geometry, kernel="bisquare", k=50, coplanar="jitter"
    )
    gw_model = GWLinearRegression(
        bandwidth=50, kernel="bisquare", fixed=False,
        keep_models=True, graph=gw_graph,
    )
    gw_model.fit(X_df, y_s, geometry=geometry)
    gw_pred = gw_model.predict(X_df, geometry=geometry)
    gw_rmse = float(np.sqrt(mean_squared_error(y_s, gw_pred)))
    gw_r2   = float(r2_score(y_s, gw_pred))

    return {
        "global_model": global_model,
        "gw_model": gw_model,
        "feature_cols": feature_cols,
        "metrics": {
            "global": {"rmse": round(global_rmse, 3), "r2": round(global_r2, 3)},
            "gwlearn": {"rmse": round(gw_rmse, 3), "r2": round(gw_r2, 3)},
        },
        "bounds": { "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y }
    }

print("Training spatial models …")
MODELS = _train_models()
print(f"  Global  RMSE={MODELS['metrics']['global']['rmse']:.3f}  R²={MODELS['metrics']['global']['r2']:.3f}")
print(f"  Spatial RMSE={MODELS['metrics']['gwlearn']['rmse']:.3f}  R²={MODELS['metrics']['gwlearn']['r2']:.3f}")
print("Models ready")

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/car.png")
def car_png():
    return send_from_directory("static", "car.png")

@app.route("/api/world")
def api_world():
    # Only send a subset of nodes for visualization if too many
    vis_nodes = WORLD_NODES
    if len(vis_nodes) > 300:
        vis_nodes = random.sample(vis_nodes, 300)
    return jsonify(nodes=vis_nodes, edges=WORLD_EDGES[:500], zone_meta=ZONE_META)

@app.route("/api/metrics")
def api_metrics():
    return jsonify(MODELS["metrics"])

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict optimal speed at a given location & hour using both models."""
    data = request.json
    x, y_coord, hour = data["x"], data["y"], data["hour"]

    # Find closest zone
    dists = [math.hypot(n["x"]-x, n["y"]-y_coord) for n in WORLD_NODES]
    closest = WORLD_NODES[int(np.argmin(dists))]
    zone = closest["zone"]
    meta = ZONE_META[zone]

    feat_arr = np.array([[hour, meta["speed"], meta["base_cong"], meta["rush_sens"]]])
    feat_df = pd.DataFrame(feat_arr, columns=MODELS["feature_cols"])
    geom = gpd.GeoSeries([Point(x, y_coord)])

    global_speed = float(MODELS["global_model"].predict(feat_arr)[0])
    gw_speed     = float(MODELS["gw_model"].predict(feat_df, geometry=geom)[0])

    return jsonify(
        zone=zone,
        global_speed=round(max(5, global_speed), 1),
        gw_speed=round(max(5, gw_speed), 1),
    )

@app.route("/api/heatmap")
def api_heatmap():
    """Return speed / congestion heatmap data for the whole map."""
    hour = float(request.args.get("hour", 8))
    
    # Use dynamic bounds from loaded world
    b = MODELS.get("bounds", {"min_x":0, "max_x":1600, "min_y":0, "max_y":800})
    
    # Grid steps proportional to world size
    w = b["max_x"] - b["min_x"]
    h = b["max_y"] - b["min_y"]
    step_x = max(50, int(w / 20))
    step_y = max(50, int(h / 20))
    
    points, feats, zones = [], [], []
    
    # Grid scan
    for gx in range(int(b["min_x"]), int(b["max_x"]), step_x):
        for gy in range(int(b["min_y"]), int(b["max_y"]), step_y):
            # heuristic: assign to nearest node's zone
            # Optimization: could use KDTree, but simple list ok for <1000 nodes
            dists = [math.hypot(n["x"]-gx, n["y"]-gy) for n in WORLD_NODES]
            closest = WORLD_NODES[int(np.argmin(dists))]
            meta = ZONE_META[closest["zone"]]
            
            points.append(Point(gx, gy))
            feats.append([hour, meta["speed"], meta["base_cong"], meta["rush_sens"]])
            zones.append(closest["zone"])

    feat_df = pd.DataFrame(feats, columns=MODELS["feature_cols"])
    geom = gpd.GeoSeries(points)
    preds = MODELS["gw_model"].predict(feat_df, geometry=geom)

    grid = []
    for i in range(len(points)):
        grid.append({
            "x": int(points[i].x), "y": int(points[i].y),
            "speed": round(max(5, float(preds[i])), 1),
            "zone": zones[i],
        })
    return jsonify(grid)



if __name__ == "__main__":
    app.run(debug=False, port=5000)
