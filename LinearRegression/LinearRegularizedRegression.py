import numpy as np

class LinearRegularizedRegression:
    """
    Base class for Ridge, Lasso, ElasticNet via gradient descent.
    
    Supports L2 (Ridge), L1 (Lasso), and Elastic Net (combination).
    Uses coordinate descent style sub-gradient for Lasso/ElasticNet.
    """
    def __init__(self,
                 alpha=1.0,           # regularization strength
                 l1_ratio=0.5,        # 0 = Ridge, 1 = Lasso, in between = ElasticNet
                 max_iter=1000,
                 tol=1e-4,
                 learning_rate=0.01,
                 momentum=0.0,        # optional Nesterov-like momentum
                 fit_intercept=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.lr = learning_rate
        self.momentum = momentum
        self.fit_intercept = fit_intercept
        
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_features_in_ = None

    def _soft_threshold(self, x, theta):
        """Proximal operator for L1 penalty (soft-thresholding)"""
        return np.sign(x) * np.maximum(np.abs(x) - theta, 0.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            self.intercept_ = y_mean
        else:
            X_c = X
            self.intercept_ = 0.0
        
        # Initialize weights
        self.coef_ = np.zeros(n_features)
        velocity = np.zeros(n_features)  # for momentum
        
        prev_loss = np.inf
        
        for iter_ in range(self.max_iter):
            # Prediction & residual
            y_pred = X_c @ self.coef_ + self.intercept_
            residual = y - y_pred
            
            # Gradient of MSE loss w.r.t. coefficients
            grad = - (X_c.T @ residual) / n_samples
            
            # Add L2 (Ridge) gradient
            l2_grad = (1 - self.l1_ratio) * self.alpha * self.coef_
            grad += l2_grad
            
            # Momentum update (Nesterov style approximation)
            velocity = self.momentum * velocity - self.lr * grad
            update = velocity
            
            # Apply update
            self.coef_ += update
            
            # Apply proximal operator for L1 part (ElasticNet / Lasso)
            l1_threshold = self.lr * self.alpha * self.l1_ratio * n_samples
            self.coef_ = self._soft_threshold(self.coef_, l1_threshold)
            
            # Check convergence (simple loss monitoring)
            current_loss = 0.5 * np.mean(residual**2) + \
                           self.alpha * ( (1-self.l1_ratio)*np.sum(self.coef_**2)/2 + \
                                          self.l1_ratio*np.sum(np.abs(self.coef_)) )
            
            if abs(prev_loss - current_loss) < self.tol:
                break
                
            prev_loss = current_loss
        
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_

    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / (ss_tot + 1e-10)


# ────────────────────────────────────────────────
#             Convenience wrapper classes
# ────────────────────────────────────────────────

class Ridge(LinearRegularizedRegression):
    """Ridge regression (pure L2 penalty)"""
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(alpha=alpha, l1_ratio=0.0, **kwargs)


class Lasso(LinearRegularizedRegression):
    """Lasso regression (pure L1 penalty)"""
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(alpha=alpha, l1_ratio=1.0, **kwargs)


class ElasticNet(LinearRegularizedRegression):
    """Elastic Net (mix of L1 + L2)"""
    def __init__(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, **kwargs)


# ────────────────────────────────────────────────
#                   Quick Example
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    # Generate correlated data (good for seeing regularization effect)
    X, y, coef_true = make_regression(n_samples=400, n_features=30,
                                      n_informative=8, noise=15,
                                      coef=True, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models = {
        "Ridge (α=1.0)": Ridge(alpha=1.0, max_iter=3000, lr=0.02, tol=1e-5),
        "Lasso (α=0.4)": Lasso(alpha=0.4, max_iter=3000, lr=0.02, tol=1e-5),
        "ElasticNet (α=0.7, ρ=0.3)": ElasticNet(alpha=0.7, l1_ratio=0.3, max_iter=3000, lr=0.02)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
        
        print(f"{name:25}  R² = {r2:.4f}   |  non-zero coefs = {n_nonzero:3d}")

    print("\nTrue number of informative features:", np.sum(np.abs(coef_true) > 0.01))