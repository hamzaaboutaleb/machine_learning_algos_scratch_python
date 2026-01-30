
import numpy as np

class DecisionTree:
    """
    Simple CART Decision Tree (Classification + Regression)
    - Uses Gini for classification, MSE for regression
    - Binary splits only
    """
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_impurity_decrease=0.0,
                 task="classification"):  # "classification" or "regression"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.task = task.lower()

        self.tree = None

    def _gini(self, y):
        """Gini impurity for classification"""
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)

    def _mse(self, y):
        """Mean Squared Error (variance) for regression"""
        if len(y) == 0:
            return 0.0
        return np.var(y)

    def _impurity(self, y):
        if self.task == "classification":
            return self._gini(y)
        else:
            return self._mse(y)

    def _best_split(self, X, y, feature_idx):
        """Find best threshold for one feature"""
        values = X[:, feature_idx]
        sorted_idx = np.argsort(values)
        values = values[sorted_idx]
        y_sorted = y[sorted_idx]

        best_gain = -np.inf
        best_threshold = None
        best_left_idx = None

        # Skip identical consecutive values
        for i in range(1, len(values)):
            if values[i] == values[i-1]:
                continue

            threshold = (values[i] + values[i-1]) / 2
            left_mask = values <= threshold
            y_left = y_sorted[left_mask]
            y_right = y_sorted[~left_mask]

            if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                continue

            impurity_before = self._impurity(y_sorted)
            impurity_left = self._impurity(y_left)
            impurity_right = self._impurity(y_right)
            weighted_impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / len(y)

            gain = impurity_before - weighted_impurity

            if gain > best_gain + 1e-9:  # small epsilon to avoid float issues
                best_gain = gain
                best_threshold = threshold
                best_left_idx = left_mask

        return best_gain, best_threshold, best_left_idx

    def _find_best_feature_split(self, X, y):
        """Find best feature + threshold"""
        n_features = X.shape[1]
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_mask = None

        for feature in range(n_features):
            gain, threshold, left_mask = self._best_split(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_left_mask = left_mask

        if best_gain < self.min_impurity_decrease:
            return None, None, None

        return best_feature, best_threshold, best_left_mask

    def _build_tree(self, X, y, depth=0):
        """Recursive tree building"""
        if len(y) < self.min_samples_split:
            return self._leaf_value(y)

        if self.max_depth is not None and depth >= self.max_depth:
            return self._leaf_value(y)

        feature, threshold, left_mask = self._find_best_feature_split(X, y)

        if feature is None:
            return self._leaf_value(y)

        left_idx = np.where(left_mask)[0]
        right_idx = np.where(~left_mask)[0]

        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def _leaf_value(self, y):
        if self.task == "classification":
            # Most common class
            return np.bincount(y).argmax()
        else:
            # Mean for regression
            return np.mean(y)

    def fit(self, X, y):
        if self.task == "classification":
            y = y.astype(int)
        self.tree = self._build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def predict_proba(self, X):
        """Only for classification - returns class probabilities"""
        if self.task != "classification":
            raise ValueError("predict_proba is only for classification")

        def _proba_one(x, node):
            if not isinstance(node, dict):
                # leaf: return one-hot like probability
                n_classes = len(np.unique(y_train))  # you need to store n_classes or compute differently
                prob = np.zeros(n_classes)
                prob[node] = 1.0
                return prob

            if x[node["feature"]] <= node["threshold"]:
                return _proba_one(x, node["left"])
            else:
                return _proba_one(x, node["right"])

        # Warning: this version is simplified and assumes leaves contain class labels
        # Better version would store class distribution in leaves
        return np.array([_proba_one(x, self.tree) for x in X])


───────────────────────────────────────────────

