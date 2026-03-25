"""
TorchANFIS Benchmark Tests

Dual-licensed under:
- GPL-3.0 (open source)
- Commercial License (proprietary use)

See LICENSE and LICENSE-COMMERCIAL for details.


| Model             | Purpose                   |
| ----------------- | ------------------------- |
| Linear Regression | sanity baseline           |
| Random Forest     | strong tabular model      |
| Gradient Boosting | strong nonlinear baseline |
| MLP               | neural baseline           |
| ClassicalANFIS    | fast, classical ANFIS     |
| ResearchANFIS     | ANFIS with modern features|

run with:
python -m benchmarks.benchmark_torchanfis
or
python benchmarks/benchmark_torchanfis.py

============================================
Example Output
============================================
Benchmark Results

            model      rmse  train_time  predict_time
3             MLP  0.507418  140.269163      0.004968
1    RandomForest  0.541697   20.305515      0.113244
2   GradientBoost  0.542233    5.001548      0.007927
5   ResearchANFIS  0.543848   14.806965      0.008209
4  ClassicalANFIS  0.615813   13.843394      0.004000
0          Linear  0.745581    0.004365      0.000000

Rule Sensitivity Results

      model      rmse  train_time  predict_time
3  ANFIS_32  0.575532    7.282908      0.005590
2  ANFIS_16  0.583327    4.310596      0.003986
1   ANFIS_8  0.611049    2.275106      0.005414
0   ANFIS_4  0.662332    2.036638      0.003531

If ANFIS is substantially worse than MLP performance, there may be:
    rule collapse
    MF saturation
    normalization instability

If ANFIS is worse than Linear Regression there is a bug in:
    rule aggregation
    normalization
    consequent solving

"""

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from torchanfis import ANFISRegressor


# =====================================================
# DATASET
# =====================================================

def load_dataset():

    data = fetch_california_housing()

    X = data.data
    y = data.target

    return train_test_split(X, y, test_size=0.2, random_state=42)


# =====================================================
# BENCHMARK RUNNER
# =====================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    start = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start

    start = time.time()

    y_pred = model.predict(X_test)

    pred_time = time.time() - start

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "model": name,
        "rmse": rmse,
        "train_time": train_time,
        "predict_time": pred_time
    }


# =====================================================
# BENCHMARK
# =====================================================

def run_benchmark():

    X_train, X_test, y_train, y_test = load_dataset()
    results = []

    models = {

        "Linear": LinearRegression(),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10
        ),

        "GradientBoost": GradientBoostingRegressor(),

        # "MLP": MLPRegressor(
            # hidden_layer_sizes=(64, 64),
            # max_iter=200
        # ),
        
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000))
        ]),

        "ClassicalANFIS": ANFISRegressor(
            n_mfs=3,
            n_rules=8,
            epochs=250
        ),

        "ResearchANFIS": ANFISRegressor(
            n_mfs=3,
            n_rules=8,
            epochs=250,
            rule_mode = "soft",
            feature_mode = "masked",
            rule_sharpening="False"
        )
    }


    for name, model in models.items():

        print(f"Running {name}...")

        res = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(res)

    df = pd.DataFrame(results)

    print("\nBenchmark Results\n")
    print(df.sort_values("rmse"))

    return df

def run_rule_sensitivity():
  
    """
    4 rules  -> underfit
    8 rules  -> good
    16 rules -> best
    32 rules -> overfit

    Diagnostics You Should Plot

    Add these plots:

    Rule firing distribution

    Detects dead rules.

    mean_firing = firing.mean(axis=0)
    Membership function collapse

    Plot MF centers:

    model.model_.mf.centers

    If they collapse to the same value → training bug.

    Feature mask sparsity
    torch.sigmoid(feature_mask)

    Should show sparse patterns.
    """

    rules = [4, 8, 16, 32]
    X_train, X_test, y_train, y_test = load_dataset()
    results = []

    for r in rules:

        model = ANFISRegressor(
            n_mfs=3,
            n_rules=r,
            epochs=50
        )

        res = evaluate_model(
            f"ANFIS_{r}",
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )
        results.append(res)

    df = pd.DataFrame(results)
    print("\nRule Sensitivity Results\n")
    print(df.sort_values("rmse"))


if __name__ == "__main__":
    run_benchmark()
    run_rule_sensitivity()
  