import numpy as np

class DecisionStump:
    """Very simple decision stump (1-level decision tree)"""
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_value = None      # value for samples <= threshold
        self.right_value = None     # value for samples > threshold

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))

        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        for feature in range(n_features):
            values = X[:, feature]
            sorted_idx = np.argsort(values)
            values = values[sorted_idx]
            y_sorted = y[sorted_idx]
            w_sorted = sample_weight[sorted_idx]

            # Try midpoints between consecutive unique values
            for i in range(1, n_samples):
                if values[i] == values[i-1]:
                    continue
                threshold = (values[i] + values[i-1]) / 2

                left_mask = values <= threshold
                w_left = w_sorted[left_mask]
                w_right = w_sorted[~left_mask]

                if len(w_left) == 0 or len(w_right) == 0:
                    continue

                # For regression stumps: mean on each side
                left_val = np.average(y_sorted[left_mask], weights=w_left)
                right_val = np.average(y_sorted[~left_mask], weights=w_right)

                # Gain = reduction in weighted variance (simple proxy)
                var_total = np.average((y_sorted - np.average(y_sorted, weights=w_sorted))**2, weights=w_sorted)
                var_left  = np.average((y_sorted[left_mask] - left_val)**2, weights=w_left) if len(w_left)>0 else 0
                var_right = np.average((y_sorted[~left_mask] - right_val)**2, weights=w_right) if len(w_right)>0 else 0
                weighted_var = (w_left.sum() * var_left + w_right.sum() * var_right) / w_sorted.sum()

                gain = var_total - weighted_var

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left_val
                    best_right = right_val

        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.left_value = best_left
        self.right_value = best_right

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        left_mask = X[:, self.feature_idx] <= self.threshold
        preds[left_mask] = self.left_value
        preds[~left_mask] = self.right_value
        return preds


class GradientBoosting:
    """
    Simple Gradient Boosting with stumps.
    
    Supports:
    - squared_error  (regression)
    - log_loss       (binary classification)
    """
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 loss='squared_error',      # 'squared_error' or 'log_loss'
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss.lower()
        self.random_state = random_state
        self.estimators = []
        self.init_pred = None           # F_0(x)
        self.is_fitted = False

    def _negative_gradient(self, y_true, y_pred):
        if self.loss == 'squared_error':
            return y_true - y_pred                  # residual = y - F_m
        elif self.loss == 'log_loss':
            # For binary log-loss: p = sigmoid(y_pred), grad = p - y
            p = 1 / (1 + np.exp(-y_pred))
            return p - y_true
        else:
            raise ValueError("Unsupported loss")

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.loss == 'log_loss':
            if not np.all(np.isin(y, [0, 1])):
                raise ValueError("log_loss requires binary targets 0/1")

        # Step 0: initial prediction (mean for regression, log-odds for classification)
        if self.loss == 'squared_error':
            self.init_pred = np.mean(y)
        else:  # log_loss → start with log-odds ≈ 0 (50% probability)
            self.init_pred = 0.0

        y_pred = np.full_like(y, self.init_pred)

        for m in range(self.n_estimators):
            # Compute negative gradient (pseudo-residuals)
            residuals = self._negative_gradient(y, y_pred)

            # Fit a stump to the residuals
            stump = DecisionStump()
            stump.fit(X, residuals)   # Note: no sample_weight here (simple version)

            # Update model: F_{m}(x) = F_{m-1}(x) + ν * h_m(x)
            update = stump.predict(X)
            y_pred += self.learning_rate * update

            self.estimators.append(stump)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = np.asarray(X, dtype=float)
        y_pred = np.full(X.shape[0], self.init_pred)

        for stump in self.estimators:
            y_pred += self.learning_rate * stump.predict(X)

        if self.loss == 'log_loss':
            # Return probabilities for classification
            return 1 / (1 + np.exp(-y_pred))
        else:
            # Raw prediction for regression
            return y_pred

    def predict_proba(self, X):
        if self.loss != 'log_loss':
            raise ValueError("predict_proba only available for log_loss")
        p = self.predict(X)
        return np.vstack([1 - p, p]).T


# ────────────────────────────────────────────────
#                   Quick Examples
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, roc_auc_score

    # ----------------------
    # Regression example
    # ----------------------
    X_reg, y_reg = make_regression(n_samples=800, n_features=12, noise=25, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    gbr = GradientBoosting(n_estimators=150, learning_rate=0.08, loss='squared_error')
    gbr.fit(X_tr, y_tr)

    y_pred_reg = gbr.predict(X_te)
    print("Regression RMSE :", np.sqrt(mean_squared_error(y_te, y_pred_reg)))

    # ----------------------
    # Binary classification
    # ----------------------
    X_clf, y_clf = make_classification(n_samples=1000, n_features=15, n_informative=8,
                                       flip_y=0.03, random_state=42)
    X_trc, X_tec, y_trc, y_tec = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    gbc = GradientBoosting(n_estimators=120, learning_rate=0.07, loss='log_loss')
    gbc.fit(X_trc, y_trc)

    proba = gbc.predict_proba(X_tec)[:, 1]
    print("Classification AUC :", roc_auc_score(y_tec, proba))