import numpy as np

class KMeans:
    """
    K-Means clustering from scratch (with k-means++ initialization)
    
    Parameters:
    -----------
    n_clusters : int, default=3
        Number of clusters (k)
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence (relative change in inertia)
    init : {'random', 'k-means++'}, default='k-means++'
        Initialization method
    random_state : int, default=None
        Seed for reproducibility
    """
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 init='k-means++', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def _init_centroids(self, X):
        """Initialize centroids"""
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        if self.init == 'k-means++':
            # k-means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            # First centroid: random point
            idx = rng.integers(0, n_samples)
            centroids[0] = X[idx]

            for k in range(1, self.n_clusters):
                # Compute squared distance to nearest centroid
                dist_sq = np.min([np.sum((X - c)**2, axis=1) for c in centroids[:k]], axis=0)
                # Choose next centroid with probability proportional to distance
                probs = dist_sq / dist_sq.sum()
                idx = rng.choice(n_samples, p=probs)
                centroids[k] = X[idx]

            return centroids

        else:
            # Random initialization
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices]

    def fit(self, X):
        """
        Fit K-Means clustering.
        X : array-like of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be larger than n_samples")

        # Initialize centroids
        self.cluster_centers_ = self._init_centroids(X)
        
        prev_inertia = np.inf
        for i in range(self.max_iter):
            # Step 1: Assign points to nearest centroid
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Step 2: Update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])

            # Compute inertia
            self.inertia_ = np.sum((X - new_centroids[labels])**2)

            # Check convergence
            if abs(prev_inertia - self.inertia_) / prev_inertia < self.tol:
                break

            self.cluster_centers_ = new_centroids
            self.labels_ = labels
            prev_inertia = self.inertia_
            self.n_iter_ = i + 1

        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        X = np.asarray(X, dtype=float)
        distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ────────────────────────────────────────────────
#                   Quick Example
# ────────────────────────────────────────────────

