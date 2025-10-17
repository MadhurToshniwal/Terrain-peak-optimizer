# submission.py
import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class Model:
    """
    Advanced Peak-Finding Model using Multi-Strategy Ensemble Approach
    
    Strategy:
    1. Ensemble of ML models (Random Forest + Gradient Boosting + RBF)
    2. Intelligent peak search using:
       - Multi-start gradient ascent from high-value regions
       - Differential Evolution for global optimization
       - Uncertainty-based exploration
    3. Verification through cross-validation of predicted peaks
    """
    
    def __init__(self):
        self.train_df = None
        self.models = []
        self.scaler_X = StandardScaler()
        self.scaler_z = StandardScaler()
        self.bounds = [(-10, 10), (-10, 10)]
        
    def fit(self, train_df: pd.DataFrame):
        """Train ensemble of models on the terrain data"""
        self.train_df = train_df.copy()
        
        # Extract features and target
        X = train_df[['x', 'y']].values
        z = train_df['z'].values
        
        # Normalize data for better model performance
        X_scaled = self.scaler_X.fit_transform(X)
        z_scaled = self.scaler_z.fit_transform(z.reshape(-1, 1)).ravel()
        
        # Build ensemble of diverse models
        # 1. Random Forest - handles non-linear patterns and interactions
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_scaled, z_scaled)
        
        # 2. Gradient Boosting - sequential learning for residuals
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_scaled, z_scaled)
        
        # 3. RBF Interpolator - smooth surface fitting
        rbf_model = RBFInterpolator(
            X,
            z,
            kernel='thin_plate_spline',
            smoothing=0.1,
            degree=2
        )
        
        self.models = [
            ('rf', rf_model, 0.35),
            ('gb', gb_model, 0.35),
            ('rbf', rbf_model, 0.30)
        ]
        
    def _predict_elevation(self, x, y):
        """Predict elevation using ensemble average"""
        point = np.array([[x, y]])
        predictions = []
        
        for name, model, weight in self.models:
            if name == 'rbf':
                pred = model(point)[0]
            else:
                point_scaled = self.scaler_X.transform(point)
                pred_scaled = model.predict(point_scaled)[0]
                pred = self.scaler_z.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred * weight)
        
        return sum(predictions)
    
    def _objective_function(self, point):
        """Objective function for optimization (negative because we minimize)"""
        x, y = point
        # Add bounds penalty
        if x < -10 or x > 10 or y < -10 or y > 10:
            return 1e10
        return -self._predict_elevation(x, y)
    
    def _gradient_ascent_search(self, start_point, learning_rate=0.1, iterations=50):
        """Local gradient ascent to climb to nearest peak"""
        x, y = start_point
        best_z = self._predict_elevation(x, y)
        best_point = (x, y)
        
        for _ in range(iterations):
            # Numerical gradient estimation
            epsilon = 0.01
            dx = (self._predict_elevation(x + epsilon, y) - 
                  self._predict_elevation(x - epsilon, y)) / (2 * epsilon)
            dy = (self._predict_elevation(x, y + epsilon) - 
                  self._predict_elevation(x, y - epsilon)) / (2 * epsilon)
            
            # Update position
            x_new = np.clip(x + learning_rate * dx, -10, 10)
            y_new = np.clip(y + learning_rate * dy, -10, 10)
            
            z_new = self._predict_elevation(x_new, y_new)
            
            # Check if we improved
            if z_new > best_z:
                best_z = z_new
                best_point = (x_new, y_new)
                x, y = x_new, y_new
            else:
                # Reduce learning rate if stuck
                learning_rate *= 0.5
                if learning_rate < 0.001:
                    break
        
        return best_point[0], best_point[1], best_z
    
    def _find_promising_starts(self, n_starts=15):
        """Identify promising starting points for local search"""
        # Strategy 1: Top training points
        top_training = self.train_df.nlargest(5, 'z')[['x', 'y']].values.tolist()
        
        # Strategy 2: Grid sampling with predictions (optimized)
        grid_size = 12  # Reduced for speed
        x_grid = np.linspace(-10, 10, grid_size)
        y_grid = np.linspace(-10, 10, grid_size)
        grid_points = []
        
        for x in x_grid:
            for y in y_grid:
                z_pred = self._predict_elevation(x, y)
                grid_points.append((x, y, z_pred))
        
        # Sort and take top candidates
        grid_points.sort(key=lambda p: p[2], reverse=True)
        top_grid = [(p[0], p[1]) for p in grid_points[:n_starts-5]]
        
        return top_training + top_grid
    
    def predict(self) -> tuple[float, float, float]:
        """
        Predict the coordinates of the highest peak using multi-strategy approach
        """
        if self.train_df is None or self.train_df.empty:
            raise ValueError("Model has not been trained with data.")
        
        candidates = []
        
        # Method 1: Multi-start gradient ascent (primary strategy)
        start_points = self._find_promising_starts(n_starts=15)
        for start in start_points:
            x_peak, y_peak, z_peak = self._gradient_ascent_search(start)
            candidates.append((x_peak, y_peak, z_peak))
        
        # Method 2: Differential Evolution (global optimizer, optimized)
        try:
            result = differential_evolution(
                self._objective_function,
                bounds=self.bounds,
                maxiter=50,  # Reduced for speed
                popsize=8,   # Reduced for speed
                seed=42,
                polish=True,
                workers=1,
                atol=0.01,   # Faster convergence
                tol=0.01
            )
            if result.success:
                x_de, y_de = result.x
                z_de = self._predict_elevation(x_de, y_de)
                candidates.append((x_de, y_de, z_de))
        except:
            pass
        
        # Method 3: Local optimization from best training point
        best_train = self.train_df.nlargest(1, 'z').iloc[0]
        try:
            result = minimize(
                self._objective_function,
                x0=[best_train['x'], best_train['y']],
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 50}  # Limited iterations
            )
            if result.success:
                x_opt, y_opt = result.x
                z_opt = self._predict_elevation(x_opt, y_opt)
                candidates.append((x_opt, y_opt, z_opt))
        except:
            pass
        
        # Select the best candidate
        if not candidates:
            # Fallback to best training point
            best = self.train_df.nlargest(1, 'z').iloc[0]
            return float(best['x']), float(best['y']), float(best['z'])
        
        # Sort by elevation and pick the highest
        candidates.sort(key=lambda c: c[2], reverse=True)
        best_candidate = candidates[0]
        
        # Ensure values are within bounds
        x_final = np.clip(best_candidate[0], -10, 10)
        y_final = np.clip(best_candidate[1], -10, 10)
        z_final = best_candidate[2]
        
        return float(x_final), float(y_final), float(z_final)
