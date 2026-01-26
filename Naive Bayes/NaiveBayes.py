import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}          # mean of each feature per class
        self.var = {}           # variance of each feature per class
        self.priors = {}        # P(class)

    def fit(self, X, y):
        """
        X : shape (n_samples, n_features)
        y : shape (n_samples,)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Calculate prior probabilities and mean/variance for each class
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / n_samples

            # Laplace smoothing for variance (avoid division by zero)
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9

    def _pdf(self, x, mean, var):
        """Gaussian probability density function"""
        exponent = np.exp(- (x - mean)**2 / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict_proba(self, X):
        """Returns probability for each class"""
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, len(self.classes)))

        # For each sample
        for i, x in enumerate(X):
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[c])
                class_conditional = np.sum(
                    np.log(self._pdf(x, self.mean[c], self.var[c]))
                )
                posteriors[i, idx] = prior + class_conditional

        # Normalize to get probabilities (log-sum-exp trick)
        log_posteriors = posteriors - np.max(posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors)
        posteriors /= np.sum(posteriors, axis=1, keepdims=True)

        return posteriors

    def predict(self, X):
        proba = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return np.array([self.classes[i] for i in class_idx])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
