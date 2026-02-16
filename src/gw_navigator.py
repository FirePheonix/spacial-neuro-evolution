"""
Geographically Weighted Navigator using GWLearn

This module implements location-aware autonomous navigation using
geographically weighted regression and classification models from GWLearn.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, Dict, List, Union
from shapely.geometry import Point

from gwlearn.linear_model import GWLinearRegression, GWLogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


class GWNavigator:
    """
    Geographically Weighted Navigation System.
    
    Uses GWLearn to create location-aware predictions for autonomous vehicle
    navigation. The key insight: optimal driving behavior varies by location!
    """
    
    def __init__(
        self,
        bandwidth: float = 0.5,
        kernel: str = 'bisquare',
        fixed: bool = False,
        adaptive: bool = True
    ):
        """
        Initialize the GW Navigator.
        
        Parameters
        ----------
        bandwidth : float
            Bandwidth for the spatial kernel (if fixed) or number of neighbors (if adaptive)
        kernel : str
            Kernel function ('gaussian', 'bisquare', 'exponential')
        fixed : bool
            Whether to use fixed bandwidth (True) or adaptive (False)
        adaptive : bool
            Use adaptive bandwidth selection via cross-validation
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.fixed = fixed
        self.adaptive = adaptive
        
        # Models will be created during fit
        self.speed_model = None
        self.acceleration_model = None
        self.maneuver_model = None
        
        # Store training data for visualization
        self.training_data = None
        self.geometry = None
        
        # Label encoder for classification
        self.label_encoder = LabelEncoder()
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_speed: np.ndarray,
        y_maneuver: Optional[np.ndarray] = None,
        geometry: Optional[gpd.GeoSeries] = None,
        verbose: bool = True
    ):
        """
        Fit geographically weighted models.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix (hour, road_class, n_lanes, etc.)
        y_speed : array
            Target: optimal speed
        y_maneuver : array, optional
            Target: optimal maneuver (categorical)
        geometry : GeoSeries, optional
            Spatial coordinates. If None, assumes X has 'x' and 'y' columns
        verbose : bool
            Print fitting progress
        """
        # Store for later use
        self.training_data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Extract geometry
        if geometry is None:
            if isinstance(X, pd.DataFrame) and 'geometry' in X.columns:
                self.geometry = X['geometry']
            elif isinstance(X, pd.DataFrame) and 'x' in X.columns and 'y' in X.columns:
                self.geometry = gpd.GeoSeries([Point(x, y) for x, y in zip(X['x'], X['y'])])
            else:
                raise ValueError("geometry must be provided or X must contain 'x' and 'y' columns")
        else:
            self.geometry = geometry
        
        # Prepare features (remove geometry columns if present)
        X_clean = self._prepare_features(X)
        
        if verbose:
            print("=" * 70)
            print("TRAINING GEOGRAPHICALLY WEIGHTED NAVIGATION MODELS")
            print("=" * 70)
            print(f"Samples: {len(X_clean)}")
            print(f"Features: {X_clean.shape[1]}")
            print(f"Bandwidth: {self.bandwidth} ({'fixed' if self.fixed else 'adaptive'})")
            print(f"Kernel: {self.kernel}")
            print()
        
        # Fit speed prediction model (regression)
        if verbose:
            print("Training GWLinearRegression for optimal speed prediction...")
        
        self.speed_model = GWLinearRegression(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            fixed=self.fixed
        )
        
        self.speed_model.fit(X_clean, y_speed, geometry=self.geometry)
        
        if verbose:
            # Evaluate on training data
            y_pred = self.speed_model.predict(X_clean, geometry=self.geometry)
            rmse = np.sqrt(mean_squared_error(y_speed, y_pred))
            r2 = r2_score(y_speed, y_pred)
            print(f"  Training RMSE: {rmse:.3f}")
            print(f"  Training R²: {r2:.3f}")
            print()
        
        # Fit maneuver classification model if provided
        if y_maneuver is not None:
            if verbose:
                print("Training GWLogisticRegression for maneuver classification...")
            
            # Encode labels
            y_maneuver_encoded = self.label_encoder.fit_transform(y_maneuver)
            
            self.maneuver_model = GWLogisticRegression(
                bandwidth=self.bandwidth,
                kernel=self.kernel,
                fixed=self.fixed
            )
            
            self.maneuver_model.fit(X_clean, y_maneuver_encoded, geometry=self.geometry)
            
            if verbose:
                y_pred = self.maneuver_model.predict(X_clean, geometry=self.geometry)
                acc = accuracy_score(y_maneuver_encoded, y_pred)
                print(f"  Training Accuracy: {acc:.3f}")
                print()
        
        if verbose:
            print("✓ Models trained successfully!")
            print("=" * 70)
        
        return self
    
    def predict_speed(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        geometry: Optional[gpd.GeoSeries] = None
    ) -> np.ndarray:
        """
        Predict optimal speed at given locations and conditions.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        geometry : GeoSeries, optional
            Spatial coordinates for predictions
            
        Returns
        -------
        speeds : ndarray
            Predicted optimal speeds
        """
        if self.speed_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X_clean = self._prepare_features(X)
        
        if geometry is None:
            if isinstance(X, pd.DataFrame) and 'geometry' in X.columns:
                geometry = X['geometry']
            elif isinstance(X, pd.DataFrame) and 'x' in X.columns and 'y' in X.columns:
                geometry = gpd.GeoSeries([Point(x, y) for x, y in zip(X['x'], X['y'])])
            else:
                raise ValueError("geometry must be provided or X must contain spatial columns")
        
        predictions = self.speed_model.predict(X_clean, geometry=geometry)
        return predictions
    
    def predict_maneuver(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        geometry: Optional[gpd.GeoSeries] = None
    ) -> np.ndarray:
        """
        Predict optimal maneuver at given locations and conditions.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        geometry : GeoSeries, optional
            Spatial coordinates for predictions
            
        Returns
        -------
        maneuvers : ndarray
            Predicted maneuvers (decoded to original labels)
        """
        if self.maneuver_model is None:
            raise ValueError("Maneuver model not fitted. Provide y_maneuver in fit().")
        
        X_clean = self._prepare_features(X)
        
        if geometry is None:
            if isinstance(X, pd.DataFrame) and 'geometry' in X.columns:
                geometry = X['geometry']
            elif isinstance(X, pd.DataFrame) and 'x' in X.columns and 'y' in X.columns:
                geometry = gpd.GeoSeries([Point(x, y) for x, y in zip(X['x'], X['y'])])
            else:
                raise ValueError("geometry must be provided or X must contain spatial columns")
        
        predictions_encoded = self.maneuver_model.predict(X_clean, geometry=geometry)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
    
    def get_local_coefficients(
        self,
        location: Point,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get location-specific model coefficients.
        
        This shows how the relationship between features and speed
        varies across geographic space!
        
        Parameters
        ----------
        location : Point
            Location to get coefficients for
        feature_names : list, optional
            Names of features
            
        Returns
        -------
        coefficients : dict
            Mapping of feature names to local coefficients
        """
        if self.speed_model is None:
            raise ValueError("Model not fitted yet.")
        
        # Get local coefficients
        # Note: GWLearn stores local coefficients during fitting
        # We'd need to refit locally or extract from stored results
        
        # For now, return the average coefficients as a placeholder
        # In practice, GWLearn computes these locally during prediction
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.speed_model.coef_))]
        
        # This is a simplified version - full implementation would
        # compute local regression at the specified location
        coefficients = {
            name: coef
            for name, coef in zip(feature_names, self.speed_model.coef_)
        }
        
        return coefficients
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test_speed: np.ndarray,
        y_test_maneuver: Optional[np.ndarray] = None,
        geometry_test: Optional[gpd.GeoSeries] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model performance on test data.
        
        Parameters
        ----------
        X_test : DataFrame
            Test features
        y_test_speed : array
            True optimal speeds
        y_test_maneuver : array, optional
            True optimal maneuvers
        geometry_test : GeoSeries, optional
            Test locations
        verbose : bool
            Print results
            
        Returns
        -------
        metrics : dict
            Evaluation metrics
        """
        results = {}
        
        # Speed prediction metrics
        y_pred_speed = self.predict_speed(X_test, geometry_test)
        results['speed_rmse'] = np.sqrt(mean_squared_error(y_test_speed, y_pred_speed))
        results['speed_mae'] = np.mean(np.abs(y_test_speed - y_pred_speed))
        results['speed_r2'] = r2_score(y_test_speed, y_pred_speed)
        
        # Maneuver classification metrics
        if y_test_maneuver is not None and self.maneuver_model is not None:
            y_pred_maneuver = self.predict_maneuver(X_test, geometry_test)
            results['maneuver_accuracy'] = accuracy_score(y_test_maneuver, y_pred_maneuver)
            results['maneuver_report'] = classification_report(
                y_test_maneuver,
                y_pred_maneuver,
                output_dict=True
            )
        
        if verbose:
            print("\n" + "=" * 70)
            print("MODEL EVALUATION RESULTS")
            print("=" * 70)
            print(f"\nSpeed Prediction (Regression):")
            print(f"  RMSE: {results['speed_rmse']:.3f} mph")
            print(f"  MAE:  {results['speed_mae']:.3f} mph")
            print(f"  R²:   {results['speed_r2']:.3f}")
            
            if 'maneuver_accuracy' in results:
                print(f"\nManeuver Classification:")
                print(f"  Accuracy: {results['maneuver_accuracy']:.3f}")
                print(f"\n  Per-class metrics:")
                for label, metrics in results['maneuver_report'].items():
                    if label in ['accuracy', 'macro avg', 'weighted avg']:
                        continue
                    print(f"    {label}:")
                    print(f"      Precision: {metrics['precision']:.3f}")
                    print(f"      Recall:    {metrics['recall']:.3f}")
                    print(f"      F1-score:  {metrics['f1-score']:.3f}")
            
            print("=" * 70 + "\n")
        
        return results
    
    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Prepare features for modeling.
        
        Removes geometry columns and handles categorical variables.
        """
        if isinstance(X, np.ndarray):
            return X
        
        X_clean = X.copy()
        
        # Remove geometry columns
        cols_to_drop = ['geometry', 'x', 'y']
        for col in cols_to_drop:
            if col in X_clean.columns:
                X_clean = X_clean.drop(columns=[col])
        
        # Handle categorical variables
        categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            X_clean = pd.get_dummies(X_clean, columns=categorical_cols, drop_first=True)
        
        return X_clean.values
    
    def navigate(
        self,
        current_location: Point,
        current_conditions: Dict,
        destination: Optional[Point] = None
    ) -> Dict:
        """
        Make navigation decision at current location and conditions.
        
        Parameters
        ----------
        current_location : Point
            Current vehicle location
        current_conditions : dict
            Current conditions (hour, road_class, etc.)
        destination : Point, optional
            Destination (for future path planning)
            
        Returns
        -------
        decision : dict
            Navigation decision with speed and maneuver
        """
        # Create feature vector
        X = pd.DataFrame([current_conditions])
        geometry = gpd.GeoSeries([current_location])
        
        # Predict
        optimal_speed = self.predict_speed(X, geometry)[0]
        
        decision = {
            'optimal_speed': optimal_speed,
            'location': current_location
        }
        
        if self.maneuver_model is not None:
            optimal_maneuver = self.predict_maneuver(X, geometry)[0]
            decision['optimal_maneuver'] = optimal_maneuver
        
        return decision


class GlobalNavigator:
    """
    Baseline: Global (non-spatial) navigation model for comparison.
    
    This assumes relationships are the same everywhere - which we know is wrong!
    """
    
    def __init__(self):
        """Initialize global navigator."""
        from sklearn.linear_model import LinearRegression, LogisticRegression
        
        self.speed_model = LinearRegression()
        self.maneuver_model = LogisticRegression(max_iter=1000)
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y_speed, y_maneuver=None, geometry=None, verbose=True):
        """Fit global models (ignore geometry)."""
        X_clean = self._prepare_features(X)
        
        if verbose:
            print("Training global (non-spatial) models...")
        
        self.speed_model.fit(X_clean, y_speed)
        
        if y_maneuver is not None:
            y_encoded = self.label_encoder.fit_transform(y_maneuver)
            self.maneuver_model.fit(X_clean, y_encoded)
        
        if verbose:
            print("✓ Global models trained")
        
        return self
    
    def predict_speed(self, X, geometry=None):
        """Predict speed (ignore geometry)."""
        X_clean = self._prepare_features(X)
        return self.speed_model.predict(X_clean)
    
    def predict_maneuver(self, X, geometry=None):
        """Predict maneuver (ignore geometry)."""
        X_clean = self._prepare_features(X)
        y_encoded = self.maneuver_model.predict(X_clean)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def evaluate(self, X_test, y_test_speed, y_test_maneuver=None, geometry_test=None, verbose=True):
        """Evaluate global model."""
        results = {}
        
        y_pred_speed = self.predict_speed(X_test)
        results['speed_rmse'] = np.sqrt(mean_squared_error(y_test_speed, y_pred_speed))
        results['speed_r2'] = r2_score(y_test_speed, y_pred_speed)
        
        if y_test_maneuver is not None:
            y_pred_maneuver = self.predict_maneuver(X_test)
            results['maneuver_accuracy'] = accuracy_score(y_test_maneuver, y_pred_maneuver)
        
        if verbose:
            print(f"\n[GLOBAL MODEL] RMSE: {results['speed_rmse']:.3f}, R²: {results['speed_r2']:.3f}")
            if 'maneuver_accuracy' in results:
                print(f"[GLOBAL MODEL] Accuracy: {results['maneuver_accuracy']:.3f}")
        
        return results
    
    def _prepare_features(self, X):
        """Prepare features (same as GWNavigator)."""
        if isinstance(X, np.ndarray):
            return X
        
        X_clean = X.copy()
        cols_to_drop = ['geometry', 'x', 'y']
        for col in cols_to_drop:
            if col in X_clean.columns:
                X_clean = X_clean.drop(columns=[col])
        
        categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            X_clean = pd.get_dummies(X_clean, columns=categorical_cols, drop_first=True)
        
        return X_clean.values


if __name__ == "__main__":
    print("GWNavigator module - use examples/ scripts for demonstrations")
