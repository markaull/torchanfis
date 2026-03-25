
"""
===============================================================================
TorchANFIS MODULE (Adaptive Neuro-Fuzzy Inference System for Python based on PyTorch library)
===============================================================================

Sklearn-style ANFIS implementation using PyTorch.

Dual-licensed under:
- GPL-3.0 (open source)
- Commercial License (proprietary use)

See LICENSE and LICENSE-COMMERCIAL for details.



FEATURES
--------
sklearn-like API:
    - fit(X, y)
    - predict(X)
    - score(X, y)
    - save(path)
    - load(path)

Training options:
    - Pure Gradient Descent
    - Hybrid Learning (Least Squares + Gradient Descent)

Membership Functions:
    - Gaussian
    - Trapezoidal

Compatible with numpy / sklearn pipelines

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
pip install torch numpy scikit-learn


-------------------------------------------------------------------------------
ANFIS Module Architecture
-------------------------------------------------------------------------------
Input Features
X  (B × F)
│
│
▼
Membership Layer
(GaussianMF or TrapezoidalMF)
│
│  Computes membership values
│
▼
mf_values
(B × F × M)
│
│
├─────────────────────────────────────────────┐
│                                             │
│                                             │
▼                                             ▼
FAST RULE ENGINE                          DIFFERENTIABLE RULE ENGINE
(Const rules)                              (Soft MF + Feature mask)

Rule indices                               Rule weights
(R × F)                                     (R × F × M)

                                            Feature masks
                                            (R × F)

│                                             │
│                                             │
│  gather / einsum                            │  softmax + einsum
│                                             │
▼                                             ▼
selected MF values
(B × R × F)

selected[b,r,f] =
membership used by rule r
for feature f
│
│
▼
Rule Aggregation
(Product T-norm)

firing = Π_f selected
│
▼
Rule firing strengths
(B × R)
│
│
▼
Normalization
│
norm = firing / Σ firing
│
▼
Normalized rule strengths
(B × R)
│
│
▼
Consequent Layer
(Sugeno Linear Models)

f_i(x) = a_i x₁ + b_i x₂ + ... + c_i

rule_outputs
(B × R)
│
│
▼
Weighted Sum
│
y = Σ norm_i * f_i
│
▼
Final Output
y_pred  (B)


-------------------------------------------------------------------------------
Tensor Shape Flow
-------------------------------------------------------------------------------
Input
X
(B × F)

Membership layer
mf_values
(B × F × M)

Rule selection
selected
(B × R × F)

Rule aggregation
firing
(B × R)

Normalization
norm
(B × R)

Consequent output
rule_outputs
(B × R)

Final output
y_pred
(B)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os # handling paths, file IO
import joblib # saving/loading XGBoost models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
# import pickle


# =============================================================================
# MEMBERSHIP FUNCTIONS
# =============================================================================

class GaussianMF(nn.Module):
    """
    Gaussian membership function:
        mu(x) = exp(-(x-c)^2 / (2*sigma^2))
    """
    def __init__(self, n_features, n_mfs):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_features, n_mfs))
        self.sigmas = nn.Parameter(torch.ones(n_features, n_mfs))

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.exp(-((x - self.centers) ** 2) /
                         (2 * self.sigmas ** 2 + 1e-6))


class TrapezoidalMF(nn.Module):
    """
    Trapezoidal membership function defined by (a,b,c,d)
    """
    def __init__(self, n_features, n_mfs):
        super().__init__()
        self.a = nn.Parameter(torch.randn(n_features, n_mfs))
        self.b = nn.Parameter(torch.randn(n_features, n_mfs))
        self.c = nn.Parameter(torch.randn(n_features, n_mfs))
        self.d = nn.Parameter(torch.randn(n_features, n_mfs))

    def forward(self, x):
        x = x.unsqueeze(-1)
        left = (x - self.a) / (self.b - self.a + 1e-6)
        right = (self.d - x) / (self.d - self.c + 1e-6)
        return torch.clamp(torch.minimum(left, right), 0, 1)


# =============================================================================
# RULE LAYER
# =============================================================================
        
# import itertools

class RuleLayer(nn.Module):
    """
    Unified rule layer supporting:

    1. Fast production ANFIS (hard MF selection)
    2. Differentiable MF assignment
    3. Per-rule feature soft masking

    The fast path is automatically used when:
        soft_mf = False
        use_feature_mask = False
    """

    def __init__(self,
                 n_features,
                 n_mfs,
                 n_rules,
                 soft_mf=False,
                 use_feature_mask=0,
                 rule_sharpening=False):

        super().__init__()

        self.n_rules = n_rules
        self.n_features = n_features
        self.n_mfs = n_mfs

        self.soft_mf = soft_mf
        self.use_feature_mask = use_feature_mask
        self.rule_sharpening = rule_sharpening

        # ------------------------------------------------------------
        # HARD RULE ASSIGNMENT (FAST PRODUCTION MODE)
        # ------------------------------------------------------------
        # if not sparse:
            # comprehensively assign MF indices per rule
            # self.rule_indices = list(itertools.product(
                # range(n_mfs), repeat=n_features
            # ))# if n_rules = n_mfs ** n_features
            
        if not soft_mf:
            # randomly assign rule_indices defines which MF each rule uses per feature
            # Shape: (n_rules, n_features)
            # Example:
            #   Rule 0 → [1,2]
            #   Rule 1 → [0,0]
            # Each rule selects exactly one MF per feature
            # Shape: (R, F)
            self.register_buffer(
                "rule_indices",
                torch.randint(0, n_mfs, (n_rules, n_features))
            )

        # ------------------------------------------------------------
        # SOFT MF ASSIGNMENT (RESEARCH MODE)
        # ------------------------------------------------------------
        else:
            # Learnable rule MF selection weights
            # Shape: (R, F, M)
            #
            # For each rule r
            #   For each feature f
            #       A distribution over M membership functions
            self.rule_weights = nn.Parameter(
                torch.randn(n_rules, n_features, n_mfs)
            )

        # ------------------------------------------------------------
        # OPTIONAL FEATURE MASKS
        # ------------------------------------------------------------
        if use_feature_mask:

            # Per-rule feature importance
            # Shape: (R, F)
            self.feature_mask = nn.Parameter(
                torch.ones(n_rules, n_features)
            )
        
        # ------------------------------------------------------------
        # rule sharpening
        # τ = 1   → normal ANFIS
        # τ > 1   → sharper rule competition
        # τ < 1   → softer rule sharing
        # ------------------------------------------------------------
        if self.rule_sharpening:
            self.log_tau = nn.Parameter(torch.zeros(1))
    
    def HardAssignmentSelect(self, mf_values):
        
        B, F, M = mf_values.shape
        
        # ------------------------------------------------------------
        # STEP 1 — Expand rule_indices to match batch size
        # ------------------------------------------------------------
        # rule_indices originally: (R, F)
        # unsqueeze(0) add a new dimension 0: (1, R, F)
        # expand(B, -1, -1) add one copy per batch element: (B, R, F)
        idx = self.rule_indices.unsqueeze(0).expand(B, -1, -1)

        # ------------------------------------------------------------
        # STEP 2 — Expand mf_values to match rule_indices size
        # ------------------------------------------------------------
        # mf_values: (B, F, M)
        # unsqueeze(1) add a new dimension 1: (B, 1, F, M)
        # expand(-1, R, -1, -1) add one copy per rule: (B, R, F, M)
        # Now for each sample (B),
        # and for each rule (R),
        # we have access to all features and all MFs.
        mf_expanded = mf_values.unsqueeze(1).expand(-1, self.n_rules, -1, -1)

        # ------------------------------------------------------------
        # STEP 3 — Select the correct MF for each feature per rule
        # ------------------------------------------------------------
        # idx: (B, R, F)
        #
        # We need to gather along the MF dimension (dim=3),
        
        # idx.unsqueeze(-1) add one extra dimension: (B, R, F, 1)
        # torch.gather(..., dim=3, index=...) index/select MF values according to rule_indices: (B, R, F, 1)
        # squeeze(-1) remove last dimension: (B, R, F)
        mf_selected = torch.gather(
            mf_expanded,
            3,
            idx.unsqueeze(-1)
        ).squeeze(-1)
        
        return mf_selected
        
    def SoftAssignmentSelect(self, mf_values):

        B, F, M = mf_values.shape

        # STEP 1 — Convert weights into probabilities; Softmax over MF dimension: (R, F, M)
        weights = torch.softmax(self.rule_weights, dim=2)

        # STEP 2 — Expand tensors for broadcasting
        # mf_values: (B, F, M) → (B, 1, F, M)
        mf_expanded = mf_values.unsqueeze(1)

        # weights: (R, F, M) → (1, R, F, M)
        weights_expanded = weights.unsqueeze(0)

        # STEP 3 — Weighted MF combination per rule; Multiply and sum over M dimension: (B, R, F)
        mf_selected = torch.sum(
            mf_expanded * weights_expanded,
            dim=3
        )
        
        return mf_selected
        
    # -------------------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------------------
    def forward(self, mf_values):
        """
        mf_values shape:
            (B, F, M)

        B = batch size
        F = number of features
        M = membership functions per feature
        """

        # ============================================================
        # FAST HARD RULE ASSIGNMENT
        # ============================================================
        if not self.soft_mf:
            mf_selected = self.HardAssignmentSelect(mf_values)

        # ============================================================
        # DIFFERENTIABLE MF SELECTION
        # ============================================================
        else:
            mf_selected = self.SoftAssignmentSelect(mf_values)

        # selected shape:
        # (B, R, F)

        # ============================================================
        # OPTIONAL FEATURE SOFT MASK
        # ============================================================
        if self.use_feature_mask == 1:
            # f_r = prod( mf_selected^a_f )
            # constrain positive; smooth and differentiable for gradient descent
            alpha = torch.nn.functional.softplus(self.feature_mask)

            log_selected = alpha.unsqueeze(0) * torch.log(mf_selected + 1e-9)
        
        # ------------------------------------------------------------
        # Multiply across features to compute rule firing
        # ------------------------------------------------------------
        # mf_selected shape: (B, R, F)
        if False:
            # torch.prod(..., dim=2) Π over features for each rule: (B, R)
            firing = torch.prod(mf_selected, dim=2)
        else:
            # log-space rule computation to avoid underflow:
            # stabilizes training when F > 10
            if self.use_feature_mask != 1:
                log_selected = torch.log(mf_selected + 1e-9)
            log_firing = torch.sum(log_selected, dim=2)
            firing = torch.exp(log_firing)
        
        # ------------------------------------------------------------
        # rule sharpening
        # τ = 1   → normal ANFIS
        # τ > 1   → sharper rule competition
        # τ < 1   → softer rule sharing
        # ------------------------------------------------------------
        if self.rule_sharpening:
            tau = torch.exp(self.log_tau)
            firing = firing ** tau

        # ------------------------------------------------------------
        # Normalize rule firing strengths
        # ------------------------------------------------------------
        # Sum across rules: (B, 1)
        # Divide each rule firing by total firing per sample.
        sum_over_rules = (firing.sum(dim=1, keepdim=True) + 1e-6)
        norm = firing / sum_over_rules

        return firing, norm

        
        
# =============================================================================
# CONSEQUENT LAYER
# =============================================================================

class ConsequentLayer(nn.Module):
    """
    First-order Sugeno consequents:
        f_i = a_i*x1 + b_i*x2 + ... + c_i
    """
    def __init__(self, n_rules, n_features):
        super().__init__()
        self.coeff = nn.Parameter(
            torch.randn(n_rules, n_features + 1)
        )

    def forward(self, x):
        ones = torch.ones(x.size(0), 1, device=x.device)
        x_aug = torch.cat([x, ones], dim=1)
        return torch.matmul(x_aug, self.coeff.T)


# =============================================================================
# ANFIS MODEL
# =============================================================================

class ANFIS(nn.Module):
    """
    Full ANFIS architecture.
    """
    def __init__(self,
            n_features,
            n_mfs=2,
            n_rules=6,
            mf_type="gaussian",
            soft_mf=False,
            use_feature_mask=0,
            rule_sharpening=False):
            
        super().__init__()

        self.n_features = n_features
        self.n_mfs = n_mfs
        self.n_rules = n_rules
        self.mf_type = mf_type

        # Membership layer
        if mf_type == "gaussian":
            self.mf_layer = GaussianMF(n_features, n_mfs)
        elif mf_type == "trapezoidal":
            self.mf_layer = TrapezoidalMF(n_features, n_mfs)
        else:
            raise ValueError("Unknown MF type")
        
        # Rule layer
        self.rule_layer = RuleLayer(
            n_features,
            n_mfs,
            n_rules,
            soft_mf=soft_mf,
            use_feature_mask=use_feature_mask,
            rule_sharpening=rule_sharpening
        )
        
        # Consequent layer
        self.consequent = ConsequentLayer(
            n_rules,
            n_features
        )

    def forward(self, x):
        mf_values = self.mf_layer(x)
        firing, norm = self.rule_layer(mf_values)
        rule_outputs = self.consequent(x)
        output = torch.sum(norm * rule_outputs, dim=1)
        return output, firing, norm


# =============================================================================
# SKLEARN-STYLE WRAPPER
# =============================================================================

class ANFISRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible ANFIS regressor.
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    def __init__(self,
                 n_mfs=2,
                 n_rules=6,
                 mf_type="gaussian",
                 rule_mode="hard",# hard | soft
                 feature_mode="full",# full | masked
                 soft_mf=None,
                 use_feature_mask=0,
                 rule_sharpening=None,
                 training="hybrid",
                 epochs=50,
                 lr=1e-2,
                 verbose=False):

        self.n_mfs = n_mfs
        self.n_rules = n_rules
        self.mf_type = mf_type
        
        self.rule_mode = rule_mode
        self.feature_mode = feature_mode

        self.soft_mf, self.use_feature_mask, self.rule_sharpening = self.resolve_modes(
            rule_mode,
            feature_mode,
            soft_mf,
            use_feature_mask,
            rule_sharpening
        )
        
        self.training = training
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.model_ =[]
        
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def resolve_modes(self, rule_mode, feature_mode, soft_mf, use_feature_mask, rule_sharpening):
        """
        Determine final configuration from modes + overrides.
        todo: switch to dictionary method if you need to extend this
        """

        # ----- rule mode presets -----
        if rule_mode == "hard":
            soft_mf_default = False
            rule_sharpening_default=False
        elif rule_mode == "soft":
            soft_mf_default = True
            rule_sharpening_default=True
        else:
            raise ValueError(f"Unknown rule_mode: {rule_mode}")

        # ----- feature mode presets -----
        if feature_mode == "full":
            feature_mask_default = 0
        elif feature_mode == "masked":
            feature_mask_default = 1
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        # ----- apply overrides -----
        if soft_mf is None:
            soft_mf = soft_mf_default
        if use_feature_mask is None:
            use_feature_mask = feature_mask_default
        if rule_sharpening is None:
            rule_sharpening = rule_sharpening_default
        return soft_mf, use_feature_mask, rule_sharpening
        
        
    def get_config(self):
        return {
            "rule_mode": self.rule_mode,
            "feature_mode": self.feature_mode,
            "soft_mf": self.soft_mf,
            "use_feature_mask": self.use_feature_mask,
            "rule_sharpening": self.rule_sharpening,
        }
    
    # -------------------------------------------------------------------------
    # MODEL CREATION
    # -------------------------------------------------------------------------
    def _build_model(self, X):
        self.model_ = ANFIS(
            n_features=X.shape[1],
            n_mfs=self.n_mfs,
            n_rules=self.n_rules,
            mf_type=self.mf_type,
            soft_mf=self.soft_mf,
            use_feature_mask=self.use_feature_mask
        )

    # -------------------------------------------------------------------------
    # HYBRID LEARNING (Least Squares for consequents)
    # -------------------------------------------------------------------------
    def _least_squares_update(self, X, y):
        self.model_.eval()

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)

            _, _, norm = self.model_(X_t)

            ones = torch.ones(X_t.size(0), 1)
            X_aug = torch.cat([X_t, ones], dim=1)

            Phi = torch.cat([
                norm[:, i:i+1] * X_aug
                for i in range(norm.shape[1])
            ], dim=1)

            theta = torch.linalg.pinv(Phi) @ y_t
            theta = theta.view(self.model_.n_rules,
                               X.shape[1] + 1)

            self.model_.consequent.coeff.data = theta

    # -------------------------------------------------------------------------
    # TRAINING FUNCTION (fit)
    # -------------------------------------------------------------------------
    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # normalize data
        X = self.x_scaler.fit_transform(X)
        # y = self.y_scaler.fit_transform(y)
        y = self.y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).squeeze()

        self._build_model(X)

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.lr
        )

        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X)
        y_t = torch.tensor(y)

        for epoch in range(self.epochs):

            # Hybrid training
            if self.training == "hybrid":
                self._least_squares_update(X, y)

            self.model_.train()
            optimizer.zero_grad()

            preds, _, _ = self.model_(X_t)
            loss = loss_fn(preds, y_t)

            loss.backward()

            # Gradient descent updates
            if self.training in ["gradient", "hybrid"]:
                optimizer.step()

            if self.verbose:
                print(f"Epoch {epoch+1}: loss={loss.item():.5f}")

        return self

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X   = self.x_scaler.transform(X)
        X_t = torch.tensor(X)

        self.model_.eval()
        with torch.no_grad():
            y_pred_sc, _, _ = self.model_(X_t)
        
        # y_pred = self.y_scaler.inverse_transform(y_pred_sc)
        y_pred = self.y_scaler.inverse_transform(
            y_pred_sc.unsqueeze(1)
        ).squeeze()
        return y_pred

    # -------------------------------------------------------------------------
    # EVALUATION (sklearn score)
    # -------------------------------------------------------------------------
    def score(self, X, y):
        X   = self.x_scaler.transform(X)
        y_pred = self.predict(X)
        rmse = root_mean_squared_error(y, y_pred) # sklearn.metrics. 
        return y_pred, rmse

    # -------------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------------
    # def save(self, path):
        # with open(path, "wb") as f:
            # pickle.dump(self, f)
            
    def save(self, path):
        """
        Save trained ANFIS model.
        """
        os.makedirs( os.path.dirname( path), exist_ok=True)
        # torch.save(self.model_.state_dict(), path)
        joblib.dump(
            {
                "model": self.model_,
                "x_scaler": self.x_scaler,
                "y_scaler": self.y_scaler,
                "n_mfs": self.n_mfs,
                "n_rules": self.n_rules,
                "mf_type": self.mf_type
            },
            path
        )

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------
    # @staticmethod
    # def load(path):
        # with open(path, "rb") as f:
            # return pickle.load(f)
    @staticmethod
    def load(path):
        """
        Load saved ANFIS model.
        """
        data = joblib.load(path)

        model = ANFISRegressor(
            n_mfs=data["n_mfs"],
            n_rules=data["n_rules"],
            mf_type=data["mf_type"]
        )

        model.model_ = data["model"]
        model.x_scaler = data["x_scaler"]
        model.y_scaler = data["y_scaler"]

        return model


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":

    # from sklearn.metrics import root_mean_squared_error
    
    # Synthetic dataset
    X = np.random.rand(200, 2)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2

    # Create model
    model = ANFISRegressor(
        n_mfs=3,
        n_rules=3,
        mf_type="gaussian",     # or "trapezoidal"
        training="hybrid",      # or "gradient"
        epochs=2,
        verbose=True
    )

    # Train
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    # rmse = root_mean_squared_error(y, y_pred) # sklearn.metrics. 
    # print( "RMSE:", rmse)


    # Evaluate
    print("R2 score:", model.score(X, y))

    # Save / Load
    model.save("C:/tmp/anfis.pkl")
    loaded = ANFISRegressor.load("C:/tmp/anfis.pkl")

    print("Loaded model score:", loaded.score(X, y))

