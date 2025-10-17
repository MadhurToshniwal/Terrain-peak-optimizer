"""
Test and Visualization Suite for Peak Finding Model

This script helps visualize the terrain, test the model performance,
and analyze the prediction accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from submission import Model
from dataset_generator import hidden_peak_function

def find_true_peak():
    """Find the actual highest peak by dense grid search"""
    x_dense = np.linspace(-10, 10, 500)
    y_dense = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x_dense, y_dense)
    Z = hidden_peak_function(X, Y)
    
    max_idx = np.argmax(Z)
    max_i, max_j = np.unravel_index(max_idx, Z.shape)
    
    true_x = X[max_i, max_j]
    true_y = Y[max_i, max_j]
    true_z = Z[max_i, max_j]
    
    return true_x, true_y, true_z

def calculate_error(pred, true):
    """Calculate horizontal and vertical errors"""
    horizontal_error = np.sqrt((pred[0] - true[0])**2 + (pred[1] - true[1])**2)
    vertical_error = abs(pred[2] - true[2])
    euclidean_error = np.sqrt(horizontal_error**2 + vertical_error**2)
    
    return {
        'horizontal': horizontal_error,
        'vertical': vertical_error,
        'euclidean': euclidean_error
    }

def visualize_terrain_and_prediction(train_df, pred, true_peak):
    """Create comprehensive visualization of the terrain and predictions"""
    fig = plt.figure(figsize=(18, 5))
    
    # Create dense grid for surface
    x_dense = np.linspace(-10, 10, 100)
    y_dense = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_dense, y_dense)
    Z = hidden_peak_function(X, Y)
    
    # 3D Surface Plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7, edgecolor='none')
    ax1.scatter(train_df['x'], train_df['y'], train_df['z'], 
                c='blue', marker='o', s=20, label='Training Data', alpha=0.6)
    ax1.scatter([pred[0]], [pred[1]], [pred[2]], 
                c='red', marker='*', s=300, label='Predicted Peak', edgecolors='black', linewidths=2)
    ax1.scatter([true_peak[0]], [true_peak[1]], [true_peak[2]], 
                c='green', marker='^', s=300, label='True Peak', edgecolors='black', linewidths=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Elevation)')
    ax1.set_title('3D Terrain with Peaks')
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour Plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='terrain')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.scatter(train_df['x'], train_df['y'], 
                c='blue', marker='o', s=30, label='Training Data', alpha=0.6)
    ax2.scatter([pred[0]], [pred[1]], 
                c='red', marker='*', s=400, label='Predicted Peak', 
                edgecolors='black', linewidths=2, zorder=5)
    ax2.scatter([true_peak[0]], [true_peak[1]], 
                c='green', marker='^', s=400, label='True Peak', 
                edgecolors='black', linewidths=2, zorder=5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Contour Map')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    
    # Heatmap
    ax3 = fig.add_subplot(133)
    im = ax3.imshow(Z, extent=[-10, 10, -10, 10], origin='lower', 
                    cmap='terrain', aspect='auto')
    ax3.scatter(train_df['x'], train_df['y'], 
                c='blue', marker='o', s=30, label='Training Data', alpha=0.6)
    ax3.scatter([pred[0]], [pred[1]], 
                c='red', marker='*', s=400, label='Predicted Peak', 
                edgecolors='black', linewidths=2, zorder=5)
    ax3.scatter([true_peak[0]], [true_peak[1]], 
                c='green', marker='^', s=400, label='True Peak', 
                edgecolors='black', linewidths=2, zorder=5)
    
    # Draw error line
    ax3.plot([pred[0], true_peak[0]], [pred[1], true_peak[1]], 
             'r--', linewidth=2, label='Error', alpha=0.7)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Heatmap with Error')
    ax3.legend()
    fig.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('terrain_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'terrain_visualization.png'")
    plt.show()

def test_model_performance(n_trials=5):
    """Test model performance across multiple random datasets"""
    results = []
    
    print(f"\n{'='*70}")
    print(f"Running {n_trials} trials with different random terrains...")
    print(f"{'='*70}\n")
    
    for trial in range(1, n_trials + 1):
        print(f"Trial {trial}/{n_trials}")
        print("-" * 50)
        
        # Generate new dataset
        from dataset_generator import generate_training_data
        train_df = generate_training_data(n_points=100)
        
        # Find true peak
        true_peak = find_true_peak()
        
        # Train model and predict
        start_time = time.time()
        model = Model()
        model.fit(train_df)
        pred = model.predict()
        elapsed_time = time.time() - start_time
        
        # Calculate errors
        errors = calculate_error(pred, true_peak)
        
        results.append({
            'trial': trial,
            'horizontal_error': errors['horizontal'],
            'vertical_error': errors['vertical'],
            'euclidean_error': errors['euclidean'],
            'time': elapsed_time,
            'pred': pred,
            'true': true_peak
        })
        
        print(f"  Predicted Peak: ({pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f})")
        print(f"  True Peak:      ({true_peak[0]:.4f}, {true_peak[1]:.4f}, {true_peak[2]:.4f})")
        print(f"  Horizontal Error: {errors['horizontal']:.4f}")
        print(f"  Vertical Error:   {errors['vertical']:.4f}")
        print(f"  Time: {elapsed_time:.2f}s")
        print()
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    df_results = pd.DataFrame(results)
    
    print(f"\nHorizontal Error (Priority Metric):")
    print(f"  Mean:   {df_results['horizontal_error'].mean():.4f}")
    print(f"  Median: {df_results['horizontal_error'].median():.4f}")
    print(f"  Min:    {df_results['horizontal_error'].min():.4f}")
    print(f"  Max:    {df_results['horizontal_error'].max():.4f}")
    
    print(f"\nVertical Error:")
    print(f"  Mean:   {df_results['vertical_error'].mean():.4f}")
    print(f"  Median: {df_results['vertical_error'].median():.4f}")
    
    print(f"\nExecution Time:")
    print(f"  Mean:   {df_results['time'].mean():.2f}s")
    print(f"  Max:    {df_results['time'].max():.2f}s")
    
    if df_results['time'].max() > 120:
        print("  ⚠️  WARNING: Some runs exceeded 2-minute time limit!")
    else:
        print("  ✅ All runs within 2-minute time limit")
    
    return df_results

def single_test_with_visualization():
    """Run a single test with full visualization"""
    print("\n" + "="*70)
    print("SINGLE TEST WITH VISUALIZATION")
    print("="*70 + "\n")
    
    # Load or generate data
    try:
        train_df = pd.read_csv('train.csv')
        print("✅ Loaded existing train.csv")
    except:
        from dataset_generator import generate_training_data
        train_df = generate_training_data()
        train_df.to_csv('train.csv', index=False)
        print("✅ Generated new train.csv")
    
    print(f"Training data: {len(train_df)} points")
    print(f"Elevation range: [{train_df['z'].min():.2f}, {train_df['z'].max():.2f}]")
    
    # Find true peak
    print("\nFinding true peak...")
    true_peak = find_true_peak()
    print(f"True peak: ({true_peak[0]:.4f}, {true_peak[1]:.4f}, {true_peak[2]:.4f})")
    
    # Train and predict
    print("\nTraining model and predicting...")
    start_time = time.time()
    model = Model()
    model.fit(train_df)
    pred = model.predict()
    elapsed_time = time.time() - start_time
    
    print(f"Predicted peak: ({pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f})")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    
    # Calculate errors
    errors = calculate_error(pred, true_peak)
    
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"Horizontal Error (Priority): {errors['horizontal']:.4f} units")
    print(f"Vertical Error:              {errors['vertical']:.4f} units")
    print(f"Euclidean Error:             {errors['euclidean']:.4f} units")
    
    # Scoring interpretation
    print(f"\n{'='*70}")
    print("PERFORMANCE ASSESSMENT")
    print(f"{'='*70}")
    if errors['horizontal'] < 0.5:
        print("🏆 EXCELLENT: Very close to true peak!")
    elif errors['horizontal'] < 1.0:
        print("✅ GOOD: Close to true peak")
    elif errors['horizontal'] < 2.0:
        print("👍 FAIR: Reasonable accuracy")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Consider tuning parameters")
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_terrain_and_prediction(train_df, pred, true_peak)
    
    return pred, true_peak, errors

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Run batch tests
        test_model_performance(n_trials=10)
    else:
        # Run single test with visualization
        single_test_with_visualization()
