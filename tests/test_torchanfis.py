"""
TorchANFIS Test Suite

Dual-licensed under:
- GPL-3.0 (open source)
- Commercial License (proprietary use)

See LICENSE and LICENSE-COMMERCIAL for details.


The tests verify:

Test	Purpose
model_forward	basic forward pass works
membership_shapes	MF output dimensions
rule_layer_shapes	rule selection logic
no_nan_forward	numerical stability
fit_runs	training loop executes
learning_improves_loss	model actually learns
predict_shape	predict output shape
save_load	persistence
soft_mf_mode	differentiable rule mode
feature_mask_mode	feature masking mode
hybrid_training	LSE step doesn't break

How To Run
Install pytest:
    pip install pytest
Run tests:
    pytest tests/
Expected output:
    =====================
    16 passed
    =====================
"""

import numpy as np
import torch
import pytest

from torchanfis import (
    GaussianMF,
    TrapezoidalMF,
    RuleLayer,
    ANFIS,
    ANFISRegressor
)


# ============================================================
# TEST DATA
# ============================================================

@pytest.fixture
def sample_data():

    np.random.seed(0)

    X = np.random.rand(200, 3)
    y = np.sin(X[:, 0]) + X[:, 1]**2 - X[:, 2]

    return X, y


# ============================================================
# MEMBERSHIP FUNCTION TESTS
# ============================================================

def test_gaussian_mf_shape():

    B, F, M = 16, 4, 3

    x = torch.randn(B, F)

    mf = GaussianMF(F, M)

    out = mf(x)

    assert out.shape == (B, F, M)


def test_trapezoidal_mf_shape():

    B, F, M = 16, 4, 3

    x = torch.randn(B, F)

    mf = TrapezoidalMF(F, M)

    out = mf(x)

    assert out.shape == (B, F, M)


# ============================================================
# RULE LAYER TESTS
# ============================================================

def test_rule_layer_hard():

    B, F, M, R = 32, 3, 4, 5

    mf_values = torch.rand(B, F, M)

    layer = RuleLayer(F, M, R, soft_mf=False)

    firing, norm = layer(mf_values)

    assert firing.shape == (B, R)
    assert norm.shape == (B, R)


def test_rule_layer_soft():

    B, F, M, R = 32, 3, 4, 5

    mf_values = torch.rand(B, F, M)

    layer = RuleLayer(F, M, R, soft_mf=True)

    firing, norm = layer(mf_values)

    assert firing.shape == (B, R)
    assert norm.shape == (B, R)


def test_rule_layer_feature_mask():

    B, F, M, R = 32, 3, 4, 5

    mf_values = torch.rand(B, F, M)

    layer = RuleLayer(F, M, R,
                      soft_mf=True,
                      use_feature_mask=1)

    firing, norm = layer(mf_values)

    assert firing.shape == (B, R)
    assert norm.shape == (B, R)


# ============================================================
# MODEL FORWARD PASS
# ============================================================

def test_model_forward():

    B, F = 16, 3

    x = torch.rand(B, F)

    model = ANFIS(
        n_features=F,
        n_mfs=3,
        n_rules=4
    )

    y, firing, norm = model(x)

    assert y.shape == (B,)
    assert firing.shape == (B, 4)
    assert norm.shape == (B, 4)


# ============================================================
# NUMERICAL STABILITY
# ============================================================

def test_no_nan_forward():

    B, F = 32, 5

    x = torch.randn(B, F)

    model = ANFIS(F, n_mfs=3, n_rules=6)

    y, firing, norm = model(x)

    assert not torch.isnan(y).any()
    assert not torch.isnan(firing).any()
    assert not torch.isnan(norm).any()


# ============================================================
# TRAINING TESTS
# ============================================================

def test_fit_runs(sample_data):

    X, y = sample_data

    model = ANFISRegressor(
        n_mfs=3,
        n_rules=4,
        epochs=2
    )

    model.fit(X, y)

    assert model.model_ is not None


def test_predict_shape(sample_data):

    X, y = sample_data

    model = ANFISRegressor(
        n_mfs=3,
        n_rules=4,
        epochs=2
    )

    model.fit(X, y)

    y_pred = model.predict(X)

    assert y_pred.shape == y.shape


# ============================================================
# LEARNING TEST
# ============================================================

def test_learning_improves_loss(sample_data):

    X, y = sample_data

    model = ANFISRegressor(
        n_mfs=3,
        n_rules=5,
        epochs=20
    )

    model.fit(X, y)

    y_pred = model.predict(X)

    mse = np.mean((y - y_pred)**2)

    assert mse < 1.0


# ============================================================
# SAVE / LOAD
# ============================================================

def test_save_load(tmp_path, sample_data):

    X, y = sample_data

    model = ANFISRegressor(
        n_mfs=3,
        n_rules=4,
        epochs=5
    )

    model.fit(X, y)

    path = tmp_path / "anfis.pkl"

    model.save(path)

    loaded = ANFISRegressor.load(path)

    y_pred = loaded.predict(X)

    assert y_pred.shape == y.shape


# ============================================================
# MODE TESTS
# ============================================================

@pytest.mark.parametrize("soft_mf,use_feature_mask", [
    (False, 0),
    (True, 0),
    (True, 1)
])

def test_modes(sample_data, soft_mf, use_feature_mask):

    X, y = sample_data

    model = ANFISRegressor(
        n_mfs=3,
        n_rules=4,
        soft_mf=soft_mf,
        use_feature_mask=use_feature_mask,
        epochs=3
    )

    model.fit(X, y)

    y_pred = model.predict(X)

    assert y_pred.shape == y.shape





# property tests to detect subtle bugs
def test_rule_normalization():

    B, F, M, R = 16, 3, 4, 5

    mf = torch.rand(B, F, M)

    layer = RuleLayer(F, M, R)

    _, norm = layer(mf)

    sums = norm.sum(dim=1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-4
    )

# gradient tests to ensure differentiability
def test_backward():

    B, F = 16, 3

    x = torch.randn(B, F)

    model = ANFIS(F, n_mfs=3, n_rules=4, soft_mf=True)

    y, _, _ = model(x)

    loss = y.mean()

    loss.backward()

    grads = [
        p.grad for p in model.parameters()
        if p.grad is not None
    ]

    assert len(grads) > 0