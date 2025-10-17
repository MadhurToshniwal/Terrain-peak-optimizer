"""
Advanced Analysis Tools for Peak Finding Model

This module provides additional analysis capabilities:
- Model performance profiling
- Feature importance analysis
- Sensitivity testing
- Ensemble contribution analysis
"""

import numpy as np
import pandas as pd
import time
from submission import Model
from dataset_generator import generate_training_data, hidden_peak_function

def profile_model_components(train_df):
    """Profile the time taken by each component of the model"""
    print("\n" + "="*70)
    print("MODEL PROFILING - COMPONENT TIMING")
    print("="*70 + "\n")
    
    model = Model()
    X = train_df[['x', 'y']].values
    z = train_df['z'].values
    
    # Profile data preprocessing
    start = time.time()
    X_scaled = model.scaler_X.fit_transform(X)
    z_scaled = model.scaler_z.fit_transform(z.reshape(-1, 1)).ravel()
    preprocess_time = time.time() - start
    print(f"Data Preprocessing: {preprocess_time:.4f}s")
    
    # Profile Random Forest
    from sklearn.ensemble import RandomForestRegressor
    start = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, 
                                min_samples_split=2, n_jobs=-1, random_state=42)
    rf.fit(X_scaled, z_scaled)
    rf_time = time.time() - start
    print(f"Random Forest Training: {rf_time:.4f}s")
    
    # Profile Gradient Boosting
    from sklearn.ensemble import GradientBoostingRegressor
    start = time.time()
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                    learning_rate=0.1, random_state=42)
    gb.fit(X_scaled, z_scaled)
    gb_time = time.time() - start
    print(f"Gradient Boosting Training: {gb_time:.4f}s")
    
    # Profile RBF
    from scipy.interpolate import RBFInterpolator
    start = time.time()
    rbf = RBFInterpolator(X, z, kernel='thin_plate_spline', 
                          smoothing=0.1, degree=2)
    rbf_time = time.time() - start
    print(f"RBF Interpolator Setup: {rbf_time:.4f}s")
    
    total_training = preprocess_time + rf_time + gb_time + rbf_time
    print(f"\nTotal Training Time: {total_training:.4f}s")
    print(f"\nTime Distribution:")
    print(f"  Random Forest: {rf_time/total_training*100:.1f}%")
    print(f"  Gradient Boosting: {gb_time/total_training*100:.1f}%")
    print(f"  RBF Interpolator: {rbf_time/total_training*100:.1f}%")
    print(f"  Preprocessing: {preprocess_time/total_training*100:.1f}%")

def analyze_ensemble_contributions(train_df, test_points=50):
    """Analyze how each model in the ensemble contributes"""
    print("\n" + "="*70)
    print("ENSEMBLE CONTRIBUTION ANALYSIS")
    print("="*70 + "\n")
    
    model = Model()
    model.fit(train_df)
    
    # Generate test points
    np.random.seed(42)
    test_x = np.random.uniform(-10, 10, test_points)
    test_y = np.random.uniform(-10, 10, test_points)
    
    # Get predictions from each model
    rf_preds = []
    gb_preds = []
    rbf_preds = []
    ensemble_preds = []
    true_vals = []
    
    for x, y in zip(test_x, test_y):
        # True value
        true_z = hidden_peak_function(x, y)
        true_vals.append(true_z)
        
        # Individual model predictions
        point = np.array([[x, y]])
        
        # RF
        point_scaled = model.scaler_X.transform(point)
        rf_pred_scaled = model.models[0][1].predict(point_scaled)[0]
        rf_pred = model.scaler_z.inverse_transform([[rf_pred_scaled]])[0, 0]
        rf_preds.append(rf_pred)
        
        # GB
        gb_pred_scaled = model.models[1][1].predict(point_scaled)[0]
        gb_pred = model.scaler_z.inverse_transform([[gb_pred_scaled]])[0, 0]
        gb_preds.append(gb_pred)
        
        # RBF
        rbf_pred = model.models[2][1](point)[0]
        rbf_preds.append(rbf_pred)
        
        # Ensemble
        ensemble_pred = model._predict_elevation(x, y)
        ensemble_preds.append(ensemble_pred)
    
    # Calculate errors
    rf_error = np.mean(np.abs(np.array(rf_preds) - np.array(true_vals)))
    gb_error = np.mean(np.abs(np.array(gb_preds) - np.array(true_vals)))
    rbf_error = np.mean(np.abs(np.array(rbf_preds) - np.array(true_vals)))
    ensemble_error = np.mean(np.abs(np.array(ensemble_preds) - np.array(true_vals)))
    
    print(f"Mean Absolute Error on {test_points} test points:")
    print(f"  Random Forest:      {rf_error:.4f}")
    print(f"  Gradient Boosting:  {gb_error:.4f}")
    print(f"  RBF Interpolator:   {rbf_error:.4f}")
    print(f"  Ensemble (Weighted): {ensemble_error:.4f}")
    
    print(f"\nError Reduction:")
    best_individual = min(rf_error, gb_error, rbf_error)
    improvement = (best_individual - ensemble_error) / best_individual * 100
    print(f"  Ensemble vs Best Individual: {improvement:+.2f}%")
    
    if ensemble_error < best_individual:
        print("\n  ✅ Ensemble outperforms all individual models!")
    else:
        print("\n  ⚠️  Ensemble doesn't improve over best individual")

def test_data_size_sensitivity():
    """Test how model performs with different amounts of training data"""
    print("\n" + "="*70)
    print("DATA SIZE SENSITIVITY ANALYSIS")
    print("="*70 + "\n")
    
    data_sizes = [25, 50, 100, 200, 400]
    results = []
    
    for n_points in data_sizes:
        print(f"Testing with {n_points} training points...")
        
        # Generate data
        train_df = generate_training_data(n_points=n_points)
        
        # Train and time
        start = time.time()
        model = Model()
        model.fit(train_df)
        pred = model.predict()
        total_time = time.time() - start
        
        # Find true peak
        x_dense = np.linspace(-10, 10, 300)
        y_dense = np.linspace(-10, 10, 300)
        X, Y = np.meshgrid(x_dense, y_dense)
        Z = hidden_peak_function(X, Y)
        max_idx = np.argmax(Z)
        max_i, max_j = np.unravel_index(max_idx, Z.shape)
        true_x, true_y = X[max_i, max_j], Y[max_i, max_j]
        
        # Calculate error
        horizontal_error = np.sqrt((pred[0] - true_x)**2 + (pred[1] - true_y)**2)
        
        results.append({
            'n_points': n_points,
            'time': total_time,
            'error': horizontal_error
        })
        
        print(f"  Time: {total_time:.2f}s, Horizontal Error: {horizontal_error:.4f}\n")
    
    # Summary
    df_results = pd.DataFrame(results)
    print("\nSummary:")
    print(df_results.to_string(index=False))
    
    print("\nRecommendation:")
    best_idx = df_results['error'].idxmin()
    best_n = df_results.loc[best_idx, 'n_points']
    print(f"  Optimal training size: {best_n} points")
    print(f"  (Balance between accuracy and speed)")

def test_terrain_complexity():
    """Test model on terrains of varying complexity"""
    print("\n" + "="*70)
    print("TERRAIN COMPLEXITY ANALYSIS")
    print("="*70 + "\n")
    
    # Define different terrain types
    terrains = {
        'Simple (Single Peak)': lambda x, y: 10 * np.exp(-((x - 3)**2 + (y - 2)**2) / 8),
        'Two Peaks': lambda x, y: (8 * np.exp(-((x - 3)**2 + (y - 2)**2) / 6) + 
                                   7 * np.exp(-((x + 4)**2 + (y + 3)**2) / 6)),
        'Complex (Original)': hidden_peak_function,
        'Noisy Ridge': lambda x, y: (5 * np.exp(-((x - 2)**2 + (y)**2) / 10) + 
                                     3 * np.sin(x) * np.cos(y) + 
                                     2 * np.random.randn())
    }
    
    for terrain_name, terrain_func in terrains.items():
        print(f"\nTesting: {terrain_name}")
        print("-" * 50)
        
        # Generate data
        np.random.seed(42)
        x_train = np.random.uniform(-10, 10, 100)
        y_train = np.random.uniform(-10, 10, 100)
        z_train = np.array([terrain_func(x, y) for x, y in zip(x_train, y_train)])
        train_df = pd.DataFrame({'x': x_train, 'y': y_train, 'z': z_train})
        
        # Find true peak
        x_dense = np.linspace(-10, 10, 300)
        y_dense = np.linspace(-10, 10, 300)
        X, Y = np.meshgrid(x_dense, y_dense)
        Z = np.array([[terrain_func(x, y) for x, y in zip(x_row, y_row)] 
                      for x_row, y_row in zip(X, Y)])
        max_idx = np.argmax(Z)
        max_i, max_j = np.unravel_index(max_idx, Z.shape)
        true_x, true_y = X[max_i, max_j], Y[max_i, max_j]
        true_z = Z[max_i, max_j]
        
        # Train and predict
        try:
            model = Model()
            model.fit(train_df)
            pred = model.predict()
            
            # Calculate error
            h_error = np.sqrt((pred[0] - true_x)**2 + (pred[1] - true_y)**2)
            v_error = abs(pred[2] - true_z)
            
            print(f"  Horizontal Error: {h_error:.4f}")
            print(f"  Vertical Error:   {v_error:.4f}")
            
            if h_error < 1.0:
                print(f"  ✅ Good performance")
            elif h_error < 2.0:
                print(f"  👍 Acceptable performance")
            else:
                print(f"  ⚠️  Challenging terrain")
        
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")

def main():
    """Run all analyses"""
    print("\n" + "="*70)
    print("ADVANCED MODEL ANALYSIS SUITE")
    print("="*70)
    
    # Generate or load training data
    try:
        train_df = pd.read_csv('train.csv')
    except:
        train_df = generate_training_data(n_points=100)
        train_df.to_csv('train.csv', index=False)
    
    # Run analyses
    profile_model_components(train_df)
    analyze_ensemble_contributions(train_df)
    test_data_size_sensitivity()
    test_terrain_complexity()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
