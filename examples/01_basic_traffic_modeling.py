"""
Example 1: Basic Spatial Traffic Modeling — with Road Network Visuals
=====================================================================

Generates a synthetic road network, simulates spatially-varying traffic,
trains Spatial vs Global models, and shows a rich matplotlib dashboard
with ACTUAL ROAD NETWORK drawings overlaid with traffic data.
"""

import sys, os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.collections as mc
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from shapely.geometry import Point
import math, random

from gwlearn.linear_model import GWLinearRegression
from libpysal.graph import Graph
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155',
    'axes.labelcolor': '#94a3b8',
    'text.color': '#e2e8f0',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'grid.color': '#334155',
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 10,
})
CYAN = '#06b6d4'; PURPLE = '#8b5cf6'; GREEN = '#22c55e'
ROAD_COLOR = '#94a3b8'; GRASS = '#1a3a25'

ZONE_META = {
    "downtown":    {"color": "#e74c3c", "base_cong": 0.7, "rush_sens": 0.9, "speed": 25},
    "commercial":  {"color": "#e67e22", "base_cong": 0.5, "rush_sens": 0.7, "speed": 35},
    "residential": {"color": "#2ecc71", "base_cong": 0.2, "rush_sens": 0.3, "speed": 35},
    "suburban":    {"color": "#3498db", "base_cong": 0.1, "rush_sens": 0.2, "speed": 45},
    "industrial":  {"color": "#9b59b6", "base_cong": 0.4, "rush_sens": 0.5, "speed": 40},
}

def build_city(n_nodes=60, n_samples=800):
    np.random.seed(42); random.seed(42)

    xs = np.random.uniform(80, 1520, n_nodes)
    ys = np.random.uniform(80, 720, n_nodes)

    # Cluster into zones
    coords = np.column_stack([xs, ys])
    labels = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(coords)
    zone_names = list(ZONE_META.keys())
    node_zones = [zone_names[l % len(zone_names)] for l in labels]

    # Build edges (connect nearby nodes)
    edges = []
    for i in range(n_nodes):
        dists = np.sqrt((xs - xs[i])**2 + (ys - ys[i])**2)
        nearby = np.argsort(dists)[1:4]
        for j in nearby:
            if dists[j] < 300:
                key = tuple(sorted((i, int(j))))
                if key not in [(e[0], e[1]) for e in edges]:
                    edges.append((key[0], key[1], float(dists[j]),
                                  random.choice(["highway", "primary", "residential"])))

    nodes_data = [(xs[i], ys[i], node_zones[i]) for i in range(n_nodes)]

    # Generate traffic samples at unique jittered locations
    rows = []
    for _ in range(n_samples):
        idx = random.randint(0, n_nodes - 1)
        x = xs[idx] + np.random.normal(0, 5)
        y = ys[idx] + np.random.normal(0, 5)
        hour = random.uniform(0, 24)
        zone = node_zones[idx]
        meta = ZONE_META[zone]

        rush = max(0, 1 - abs(hour - 8)/3) + max(0, 1 - abs(hour - 17.5)/3)
        loc_mod = math.sin(x / 200) * math.cos(y / 200)
        cong = meta["base_cong"] + meta["rush_sens"] * rush * (1 + 0.3 * loc_mod)
        cong = max(0, min(1, cong))
        speed = meta["speed"] * (1 - cong * (0.5 + 0.2 * loc_mod))
        speed = max(5, speed + np.random.normal(0, 2))

        rows.append({"x": x, "y": y, "hour": hour,
                      "speed_limit": meta["speed"],
                      "base_congestion": meta["base_cong"],
                      "rush_sensitivity": meta["rush_sens"],
                      "optimal_speed": speed, "zone": zone})

    df = pd.DataFrame(rows)
    geometry = gpd.GeoSeries([Point(r["x"], r["y"]) for r in rows])
    return df, geometry, nodes_data, edges, xs, ys


def draw_road_network(ax, xs, ys, edges, node_zones, alpha=1.0):
    """Draw the actual road network on a matplotlib axis — roads as thick grey lines."""
    # Green grass background
    ax.set_facecolor(GRASS)

    # Draw edges as thick grey roads with white borders
    for u, v, length, road_class in edges:
        lw = {"highway": 6, "primary": 4, "residential": 3}.get(road_class, 3)
        # White border
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]],
                color='white', linewidth=lw + 2, alpha=0.3 * alpha, solid_capstyle='round')
        # Grey road
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]],
                color='#555555', linewidth=lw, alpha=0.7 * alpha, solid_capstyle='round')
        # Dashed center line
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]],
                color='white', linewidth=0.5, alpha=0.4 * alpha,
                linestyle='--', dash_capstyle='round')

    # Draw zone-colored nodes (intersections)
    for i, (x, y, z) in enumerate(zip(xs, ys, node_zones)):
        ax.plot(x, y, 'o', color=ZONE_META[z]["color"], markersize=4,
                markeredgecolor='white', markeredgewidth=0.5, alpha=0.8 * alpha)


def main():
    print("=" * 70)
    print("  SPATIOTEMPORAL TRAFFIC MODELING — Spatial vs Global")
    print("  With Road Network Visualization")
    print("=" * 70)

    # ── 1. Build city ───────────────────────────────────────────────
    print("\n[1/6] Building road network & generating traffic...")
    df, geometry, nodes_data, edges, xs, ys = build_city()
    node_zones = [n[2] for n in nodes_data]
    print(f"      {len(xs)} nodes, {len(edges)} roads, {len(df)} traffic samples")

    # ── 2. Features ─────────────────────────────────────────────────
    print("[2/6] Preparing features (NO zone labels — fair test)...")
    feat_cols = ["hour", "speed_limit", "base_congestion", "rush_sensitivity"]
    X = df[feat_cols].astype(float)
    y = pd.Series(df["optimal_speed"].values, name="optimal_speed")

    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=42)
    X_tr = X.iloc[idx_tr].reset_index(drop=True)
    X_te = X.iloc[idx_te].reset_index(drop=True)
    y_tr = y.iloc[idx_tr].reset_index(drop=True)
    y_te = y.iloc[idx_te].reset_index(drop=True)
    g_tr = gpd.GeoSeries([geometry.iloc[i] for i in idx_tr]).reset_index(drop=True)
    g_te = gpd.GeoSeries([geometry.iloc[i] for i in idx_te]).reset_index(drop=True)
    zones_te = df["zone"].iloc[idx_te].values
    print(f"      Train: {len(X_tr)}  Test: {len(X_te)}")

    # ── 3. Global Model ─────────────────────────────────────────────
    print("[3/6] Training Global baseline...")
    gl = LinearRegression().fit(X_tr.values, y_tr.values)
    yp_gl = gl.predict(X_te.values)
    gl_rmse = float(np.sqrt(mean_squared_error(y_te, yp_gl)))
    gl_r2 = float(r2_score(y_te, yp_gl))
    print(f"      Global  — RMSE: {gl_rmse:.3f}  R²: {gl_r2:.3f}")

    # ── 4. GWLearn Model ────────────────────────────────────────────
    print("[4/6] Training Spatial Model (spatially-aware)...")
    graph = Graph.build_kernel(g_tr, kernel="bisquare", k=20, coplanar="jitter")
    gw = GWLinearRegression(
        bandwidth=20, kernel="bisquare", fixed=False,
        keep_models=True, graph=graph,
    )
    gw.fit(X_tr, y_tr, geometry=g_tr)
    yp_gw = np.array(gw.predict(X_te, geometry=g_te))
    gw_rmse = float(np.sqrt(mean_squared_error(y_te, yp_gw)))
    gw_r2 = float(r2_score(y_te, yp_gw))
    print(f"      Spatial — RMSE: {gw_rmse:.3f}  R²: {gw_r2:.3f}")

    imp = (gl_rmse - gw_rmse) / gl_rmse * 100
    print(f"\n  Spatial Model is {imp:.1f}% more accurate!\n")

    # Errors
    e_gl = np.abs(y_te.values - yp_gl)
    e_gw = np.abs(y_te.values - yp_gw)

    # ── 5. Dashboard ────────────────────────────────────────────────
    print("[5/6] Creating dashboard with road network...")
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle('Spatiotemporal AV Navigator — Spatial vs Global',
                 fontsize=16, fontweight='bold', color=CYAN, y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.32, wspace=0.28,
                           left=0.05, right=0.97, top=0.92, bottom=0.06)

    # ── Panel 1: Road network + traffic speed overlay ──
    ax1 = fig.add_subplot(gs[0, 0])
    draw_road_network(ax1, xs, ys, edges, node_zones, alpha=0.6)
    sc1 = ax1.scatter(df["x"], df["y"], c=df["optimal_speed"], s=8,
                      alpha=0.8, cmap='RdYlGn', edgecolors='none', zorder=5)
    plt.colorbar(sc1, ax=ax1, label='Speed (mph)', shrink=0.75)
    ax1.set_title('Road Network + Traffic Speed', fontweight='bold', color='white')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')
    ax1.set_aspect('equal')

    # ── Panel 2: Zone map with legend ──
    ax2 = fig.add_subplot(gs[0, 1])
    draw_road_network(ax2, xs, ys, edges, node_zones, alpha=0.3)
    for zone, meta in ZONE_META.items():
        mask = df["zone"] == zone
        ax2.scatter(df.loc[mask, "x"], df.loc[mask, "y"], s=10, alpha=0.7,
                    color=meta["color"], label=f'{zone} ({meta["speed"]}mph)', zorder=5)
    ax2.legend(fontsize=7, loc='lower right', framealpha=0.7,
               facecolor='#1e293b', edgecolor='#334155')
    ax2.set_title('Traffic Zones', fontweight='bold', color='white')
    ax2.set_aspect('equal')

    # ── Panel 3: Predicted vs Actual ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_te, yp_gl, s=10, alpha=0.35, color=PURPLE, label=f'Global R²={gl_r2:.2f}')
    ax3.scatter(y_te, yp_gw, s=10, alpha=0.35, color=CYAN, label=f'Spatial R²={gw_r2:.2f}')
    lims = [y_te.min()-2, y_te.max()+2]
    ax3.plot(lims, lims, '--', color='#475569', lw=1.5)
    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.set_xlabel('Actual Speed'); ax3.set_ylabel('Predicted Speed')
    ax3.set_title('Predicted vs Actual', fontweight='bold', color='white')
    ax3.legend(fontsize=8)

    # ── Panel 4: Error improvement map on road network ──
    ax4 = fig.add_subplot(gs[1, 0])
    draw_road_network(ax4, xs, ys, edges, node_zones, alpha=0.3)
    diff = e_gl - e_gw
    sc4 = ax4.scatter(g_te.x, g_te.y, c=diff, s=18, alpha=0.8,
                      cmap='RdBu', vmin=-5, vmax=5, edgecolors='none', zorder=5)
    plt.colorbar(sc4, ax=ax4, label='Error Diff (mph)', shrink=0.75)
    ax4.set_title('Spatial Advantage Map (Blue=Better)', fontweight='bold', color='white')
    ax4.set_aspect('equal')

    # ── Panel 5: Error by zone bar chart ──
    ax5 = fig.add_subplot(gs[1, 1])
    zone_errs = pd.DataFrame({"zone": zones_te, "global": e_gl, "gwlearn": e_gw})
    means = zone_errs.groupby("zone").mean()
    x_pos = np.arange(len(means))
    b1 = ax5.bar(x_pos - 0.2, means["global"], 0.35, color=PURPLE, alpha=0.8, label="Global")
    b2 = ax5.bar(x_pos + 0.2, means["gwlearn"], 0.35, color=CYAN, alpha=0.8, label="Spatial")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(means.index, rotation=30, fontsize=8)
    ax5.set_ylabel('Mean Abs Error (mph)')
    ax5.set_title('Error by Zone', fontweight='bold', color='white')
    ax5.legend(fontsize=8)

    # ── Panel 6: Scoreboard ──
    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis('off')
    for val, label, col, yy in [
        (f"{gl_rmse:.2f}", "Global RMSE", PURPLE, 0.75),
        (f"{gw_rmse:.2f}", "Spatial RMSE", CYAN, 0.50),
        (f"{imp:+.1f}%", "RMSE Improvement", GREEN if imp > 0 else '#ef4444', 0.25),
    ]:
        ax6.text(0.5, yy+0.06, val, fontsize=30, fontweight='bold', color=col,
                 ha='center', transform=ax6.transAxes)
        ax6.text(0.5, yy-0.04, label, fontsize=10, color='#94a3b8',
                 ha='center', transform=ax6.transAxes)
    ax6.set_title('Scoreboard', fontweight='bold', color='white')

    fig.savefig(os.path.join(vis_dir, 'dashboard.png'), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    print("      ✓ Saved visualizations/dashboard.png")

    # ── 6. Rush hour ────────────────────────────────────────────────
    print("[6/6] Creating rush hour comparison...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Rush Hour Impact — Spatial Non-Stationarity',
                  fontsize=14, fontweight='bold', color='#f97316', y=1.02)
    for i, (lo, hi, lbl) in enumerate([
        (6, 9, 'Morning Rush (6-9h)'),
        (12, 14, 'Midday (12-14h)'),
        (17, 19, 'Evening Rush (17-19h)')
    ]):
        ax = axes2[i]
        draw_road_network(ax, xs, ys, edges, node_zones, alpha=0.4)
        mask = df["hour"].between(lo, hi)
        sub = df[mask]
        sc = ax.scatter(sub["x"], sub["y"], c=sub["optimal_speed"],
                        s=18, alpha=0.85, cmap='RdYlGn', vmin=5, vmax=50,
                        edgecolors='none', zorder=5)
        ax.set_title(lbl, fontweight='bold', color='white')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, shrink=0.75)
    fig2.tight_layout()
    fig2.savefig(os.path.join(vis_dir, 'rush_hour.png'), dpi=150,
                 facecolor=fig2.get_facecolor(), bbox_inches='tight')
    print("      ✓ Saved visualizations/rush_hour.png")

    plt.show()

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print(f"  Global  RMSE={gl_rmse:.2f}  R²={gl_r2:.3f}")
    print(f"  Spatial RMSE={gw_rmse:.2f}  R²={gw_r2:.3f}")
    print(f"   Spatial: {imp:+.1f}% improvement")
    print("=" * 70)


if __name__ == "__main__":
    main()
