"""
Example 2: Evolving Spatial Intelligence
========================================

Uses a Genetic Algorithm to evolve the optimal spatial bandwidth.
Each "brain" is a Spatial model with a different bandwidth gene.
They compete on prediction accuracy. Best brains reproduce with mutation.

Key: We do NOT pass a pre-built graph. Each brain builds its own
spatial weights using its unique bandwidth, so bandwidth actually matters.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from gwlearn.linear_model import GWLinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.filterwarnings("ignore")

def generate_city(n=150):
    print(f"  Generating city with {n} traffic sensors...")
    np.random.seed(42)
    random.seed(42)

    xs = np.random.uniform(0, 1000, n)
    ys = np.random.uniform(0, 1000, n)
    hours = np.random.uniform(0, 24, n)

    # Add tiny jitter to avoid perfectly coplanar points
    xs += np.random.normal(0, 0.01, n)
    ys += np.random.normal(0, 0.01, n)

    speeds = []
    for i in range(n):
        x, y, h = xs[i], ys[i], hours[i]
        # SPATIAL rule: center is fast, edges slow
        dist_center = np.sqrt((x - 500)**2 + (y - 500)**2)
        base = 55 - dist_center * 0.06

        # TEMPORAL rule: rush hour penalty (stronger near edges)
        rush = max(0, 1 - abs(h - 8) / 3) + max(0, 1 - abs(h - 17.5) / 3)
        penalty = rush * (10 + dist_center * 0.02)

        # SPATIAL interaction: NW quadrant has construction (extra slow)
        if x < 400 and y < 400:
            penalty += 8

        speed = max(5, base - penalty + np.random.normal(0, 2))
        speeds.append(speed)

    df = pd.DataFrame({
        "x": xs, "y": ys, "hour": hours,
        "dist_center": np.sqrt((xs - 500)**2 + (ys - 500)**2),
        "optimal_speed": speeds,
    })
    geometry = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)])
    return df, geometry

class SpatialBrain:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth or random.uniform(50, 800)
        self.fitness = 0.0
        self.rmse = float("inf")

    def evaluate(self, X_train, y_train, geom_train, X_test, y_test, geom_test):
        """
        Train on train set, score on TEST set (out-of-sample).
        Each brain builds its OWN graph — bandwidth actually matters.
        """
        try:
            model = GWLinearRegression(
                bandwidth=self.bandwidth,
                kernel="bisquare",
                fixed=True,
                keep_models=True,
            )
            model.fit(X_train, y_train, geometry=geom_train)
            preds = model.predict(X_test, geometry=geom_test)
            self.rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            self.fitness = 1.0 / (self.rmse + 1e-6)
        except Exception as e:
            print(f"    Warning: bw={self.bandwidth:.0f} error: {e}")
            self.rmse = 9999
            self.fitness = 0
        return self.rmse

def run_evolution():
    df, geometry = generate_city()

    feature_cols = ["hour", "dist_center"]
    X = df[feature_cols].copy()
    y = pd.Series(df["optimal_speed"].values, name="optimal_speed")

    # Train/Test split so fitness reflects GENERALIZATION, not memorization
    idx_train, idx_test = train_test_split(range(len(df)), test_size=0.3, random_state=42)
    X_train, X_test = X.iloc[idx_train].reset_index(drop=True), X.iloc[idx_test].reset_index(drop=True)
    y_train, y_test = y.iloc[idx_train].reset_index(drop=True), y.iloc[idx_test].reset_index(drop=True)
    geom_train = gpd.GeoSeries([geometry.iloc[i] for i in idx_train]).reset_index(drop=True)
    geom_test  = gpd.GeoSeries([geometry.iloc[i] for i in idx_test]).reset_index(drop=True)

    POP_SIZE = 6
    NUM_GENS = 5
    ELITE = 2

    # Spawn initial population with diverse bandwidths
    population = [SpatialBrain(bw) for bw in [50, 150, 300, 500, 700, 900]]

    print(f"\n Starting evolution: {POP_SIZE} brains x {NUM_GENS} generations")
    print(f"   Train: {len(idx_train)} points, Test: {len(idx_test)} points\n")

    for gen in range(1, NUM_GENS + 1):
        print(f"═══ Generation {gen} ═══")

        for i, brain in enumerate(population):
            rmse = brain.evaluate(X_train, y_train, geom_train, X_test, y_test, geom_test)
            bar_len = max(1, int((20 - rmse) * 3))
            bar = "█" * max(1, min(40, bar_len))
            print(f"  Brain {i+1}  bw={brain.bandwidth:6.0f}  "
                  f"RMSE={rmse:6.2f}  {bar}")

        population.sort(key=lambda b: b.fitness, reverse=True)
        best = population[0]
        worst = population[-1]
        print(f"   Best:  bw={best.bandwidth:.0f}  RMSE={best.rmse:.2f}")
        print(f"   Worst: bw={worst.bandwidth:.0f}  RMSE={worst.rmse:.2f}\n")

        if gen == NUM_GENS:
            break

        # --- Reproduce ---
        elites = population[:ELITE]
        children = [SpatialBrain(e.bandwidth) for e in elites]  # keep elites

        while len(children) < POP_SIZE:
            parent = random.choice(elites)
            # Mutate bandwidth by ±10-20%
            factor = 1 + np.random.normal(0, 0.15)
            child_bw = max(20, parent.bandwidth * factor)
            children.append(SpatialBrain(bandwidth=child_bw))

        population = children

    print("━" * 50)
    print(f" EVOLUTION COMPLETE")
    print(f"   Evolved bandwidth : {best.bandwidth:.0f} distance units")
    print(f"   Final test RMSE   : {best.rmse:.2f}")
    print(f"")
    print(f"   Small bw = hyper-local (overfits noise)")
    print(f"   Large bw = too global (misses local patterns)")
    print(f"   Evolution found the sweet spot!")
    print("━" * 50)


if __name__ == "__main__":
    run_evolution()
