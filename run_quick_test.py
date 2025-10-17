"""
Quick Runner Script - Bluestone Peak Finding Challenge

This script demonstrates the complete workflow:
1. Generate/load training data
2. Train the model
3. Make prediction
4. Display results
"""

import pandas as pd
import time
from submission import Model

def main():
    print("\n" + "="*70)
    print("🏔️  BLUESTONE PEAK FINDING CHALLENGE - QUICK TEST")
    print("="*70 + "\n")
    
    # Step 1: Load or generate data
    print("Step 1: Loading training data...")
    try:
        train_df = pd.read_csv('train.csv')
        print(f"✅ Loaded train.csv with {len(train_df)} data points")
    except FileNotFoundError:
        print("⚠️  train.csv not found. Generating new dataset...")
        from dataset_generator import generate_training_data
        train_df = generate_training_data(n_points=100)
        train_df.to_csv('train.csv', index=False)
        print(f"✅ Generated and saved train.csv with {len(train_df)} data points")
    
    print(f"   Coordinate range: x=[{train_df['x'].min():.2f}, {train_df['x'].max():.2f}], "
          f"y=[{train_df['y'].min():.2f}, {train_df['y'].max():.2f}]")
    print(f"   Elevation range: z=[{train_df['z'].min():.2f}, {train_df['z'].max():.2f}]")
    print(f"   Highest training point: z={train_df['z'].max():.2f}")
    
    # Step 2: Initialize and train model
    print("\nStep 2: Training ensemble model...")
    print("   - Random Forest (100 trees)")
    print("   - Gradient Boosting (100 estimators)")
    print("   - RBF Interpolator (thin plate spline)")
    
    start_time = time.time()
    model = Model()
    model.fit(train_df)
    train_time = time.time() - start_time
    print(f"✅ Model trained in {train_time:.2f} seconds")
    
    # Step 3: Predict the peak
    print("\nStep 3: Searching for the highest peak...")
    print("   Strategy: Multi-start gradient ascent + global optimization")
    
    predict_start = time.time()
    x_pred, y_pred, z_pred = model.predict()
    predict_time = time.time() - predict_start
    
    total_time = time.time() - start_time
    
    # Step 4: Display results
    print("\n" + "="*70)
    print("🎯 PREDICTION RESULTS")
    print("="*70)
    print(f"\n   Predicted Peak Location:")
    print(f"   ├─ X coordinate: {x_pred:>10.4f}")
    print(f"   ├─ Y coordinate: {y_pred:>10.4f}")
    print(f"   └─ Z elevation:  {z_pred:>10.4f}")
    
    print(f"\n   Performance Metrics:")
    print(f"   ├─ Training time:    {train_time:>6.2f}s")
    print(f"   ├─ Prediction time:  {predict_time:>6.2f}s")
    print(f"   └─ Total time:       {total_time:>6.2f}s")
    
    # Time limit check
    if total_time < 120:
        print(f"\n   ✅ Within 2-minute time limit ({total_time:.2f}s / 120s)")
    else:
        print(f"\n   ⚠️  Exceeds 2-minute time limit ({total_time:.2f}s / 120s)")
    
    # Compare with best training point
    best_train = train_df.nlargest(1, 'z').iloc[0]
    improvement = z_pred - best_train['z']
    
    print(f"\n   Comparison to Best Training Point:")
    print(f"   ├─ Best training elevation: {best_train['z']:.4f}")
    print(f"   ├─ Predicted elevation:     {z_pred:.4f}")
    print(f"   └─ Improvement:             {improvement:+.4f}")
    
    if improvement > 0:
        print(f"\n   🏆 Model found a higher peak than any training point!")
    elif improvement > -0.5:
        print(f"\n   ✅ Prediction is competitive with training data")
    
    # Recommendations
    print("\n" + "="*70)
    print("📊 NEXT STEPS")
    print("="*70)
    print("\n   To evaluate model performance more thoroughly:")
    print("   1. Run: python test_model.py")
    print("      → Generates 3D visualization and error analysis")
    print("\n   2. Run: python test_model.py --batch")
    print("      → Tests on 10 different random terrains")
    print("\n   3. Modify dataset_generator.py to create different terrains")
    print("      → Test model robustness across terrain types")
    
    print("\n" + "="*70 + "\n")
    
    return x_pred, y_pred, z_pred

if __name__ == "__main__":
    try:
        result = main()
        print("✅ Quick test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
