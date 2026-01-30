import numpy as np

class PCA:
    """
    Principal Component Analysis from scratch using NumPy.
    
    Parameters:
    -----------
    n_components : int or float, default=None
        - If int: number of components to keep
        - If float (0 < n_components < 1): keep components that explain this much variance
    whiten : bool, default=False
        Whether to scale components to unit variance
    """
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        
        self.mean_ = None
        self.components_ = None       # principal directions (eigenvectors)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_features_in_ = None
        self.n_samples_ = None

    def fit(self, X):
        """
        Fit the PCA model.
        X : array-like of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        self.n_samples_, self.n_features_in_ = X.shape
        
        # Step 1: Center the data (subtract mean)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)  # shape (n_features, n_features)
        
        # Step 3: Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Step 4: Sort by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep only real parts (numerical stability)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Explained variance & ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / total_var
        
        # Decide how many components to keep
        if self.n_components is None:
            self.n_components = self.n_features_in_
        elif 0 < self.n_components < 1:
            # Find minimal k such that explained variance >= target
            cum_var = np.cumsum(self.explained_variance_ratio_)
            self.n_components = np.argmax(cum_var >= self.n_components) + 1
        elif isinstance(self.n_components, int):
            if self.n_components > self.n_features_in_:
                self.n_components = self.n_features_in_
        else:
            raise ValueError("n_components must be int or float in (0,1)")
        
        # Step 5: Store top components
        self.components_ = eigenvectors[:, :self.n_components].T  # shape (n_components, n_features)
        
        # Optional whitening: scale eigenvectors by 1/sqrt(eigenvalue)
        if self.whiten:
            scale = 1.0 / np.sqrt(eigenvalues[:self.n_components] + 1e-10)
            self.components_ *= scale[:, np.newaxis]
        
        return self

    def transform(self, X):
        """
        Project data onto principal components.
        Returns shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct original data (approximate)
        """
        return X_transformed @ self.components_ + self.mean_

    def get_explained_variance(self):
        return self.explained_variance_[:self.n_components]

    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio_[:self.n_components]

    def get_cumulative_explained_variance_ratio(self):
        return np.cumsum(self.explained_variance_ratio_[:self.n_components])


# ────────────────────────────────────────────────
#                   Quick Example
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Standardize (very important for PCA!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply our PCA
    pca = PCA(n_components=0.95)          # keep enough to explain ≥95% variance
    X_pca = pca.fit_transform(X_scaled)

    print(f"Original shape: {X.shape}")
    print(f"Reduced shape : {X_pca.shape}")
    print(f"Explained variance ratio: {pca.get_explained_variance_ratio()}")
    print(f"Cumulative explained variance: {pca.get_cumulative_explained_variance_ratio()}")

    # Visualize first two components
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=60)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("Iris dataset — PCA projection")
    plt.colorbar(scatter, ticks=range(3), label="Species")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Scree plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
             np.cumsum(pca.explained_variance_ratio_),
             marker='o', linestyle='--')
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Explained Variance by Principal Components")
    plt.grid(True)
    plt.show()