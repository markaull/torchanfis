# TorchANFIS

Adaptive Neuro-Fuzzy Inference System (ANFIS) implemented in PyTorch with a clean, sklearn-style API.

---

## Features

### Sklearn-style interface:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`
- `save(path)` / `load(path)`

### Training modes:
- Gradient Descent
- Hybrid Learning (Least Squares + Gradient)

### Advanced capabilities:
- Hard (classical) rule selection
- Soft (differentiable) rule selection
- Feature masking (learn feature importance per rule)
- Rule sharpening (temperature scaling)

### Compatible with:
- NumPy
- Scikit-learn pipelines


## Architecture

Input → Membership Functions → Rule Layer → Normalization → Consequents → Output


### Key Components

| Layer              | Description |
|-------------------|------------|
| Membership Layer  | Gaussian / Trapezoidal fuzzy sets |
| Rule Layer        | Hard or differentiable rule selection |
| Consequent Layer  | First-order Sugeno linear models |


### Research Modes
Rule Modes
Mode	Description
hard	classical ANFIS
soft	differentiable rule selection

Feature Modes
Mode	Description
full	all features used
masked	learn feature importance per rule


## Usage
## Quick Example
import numpy as np
from torchanfis import ANFISRegressor

### Synthetic data
X = np.random.rand(200, 2)
y = np.sin(X[:, 0]) + X[:, 1] ** 2

model = ANFISRegressor(
    n_mfs=3,
    n_rules=4,
    epochs=50
)

model.fit(X, y)

y_pred = model.predict(X)

print("RMSE:", model.score(X, y))


### Testing

Install pytest:

pip install pytest

Run tests:

pytest tests/

Expected:

=====================
16 passed
=====================


### Test Coverage
Test	Purpose
model_forward	forward pass correctness
membership_shapes	MF outputs
rule_layer_shapes	rule logic
no_nan_forward	numerical stability
fit_runs	training loop
learning_improves_loss	model learns
predict_shape	output shape
save_load	persistence
soft_mf_mode	differentiable rules
feature_mask_mode	feature masking
hybrid_training	LSE step

### Benchmark

Run:

python -m benchmarks.benchmark_torchanfis

or:

python benchmarks/benchmark_torchanfis.py
Models Compared
- Linear Regression

Random Forest

Gradient Boosting

MLP

Classical ANFIS

Research ANFIS

Example Output
model              rmse
MLP                0.50
RandomForest       0.54
GradientBoost      0.54
ResearchANFIS      0.54
ClassicalANFIS     0.61
Linear             0.74


### Save / Load
model.save("model.pkl")
loaded = ANFISRegressor.load("model.pkl")

## License

This project is dual-licensed:
- Open-source: GPL-3.0
- Commercial: available on request

You may use this project under the terms of the GPL-3.0 license, or obtain a commercial license for proprietary use.

For commercial licensing, contact:
mark.aull+torchanfis@gmail.com
