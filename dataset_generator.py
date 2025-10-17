import numpy as np
import pandas as pd

# === CONFIG ===
N_POINTS = 100
X_RANGE = (-10, 10)
Y_RANGE = (-10, 10)
NOISE_STD = 0.3

def hidden_peak_function(x, y):
    z = (
        8 * np.exp(-((x - 2.73)**2 + (y + 4.58)**2) / 6) +
        10 * np.exp(-((x + 5.64)**2 + (y - 3.29)**2) / 8) +
        4 * np.sin(0.7 * x) * np.cos(0.5 * y)
    )
    return z

# === Data generation ===
def generate_training_data(n_points=N_POINTS):
    x = np.random.uniform(X_RANGE[0], X_RANGE[1], n_points)
    y = np.random.uniform(Y_RANGE[0], Y_RANGE[1], n_points)
    z = hidden_peak_function(x, y) + np.random.normal(0, NOISE_STD, n_points)
    df = pd.DataFrame({"x": x.round(2), "y": y.round(2), "z": z.round(2)})
    return df

if __name__ == "__main__":
    train_df = generate_training_data()
    train_df.to_csv("train.csv", index=False)
    print("✅ Training dataset saved as train.csv")
