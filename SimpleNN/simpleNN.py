import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class SimpleNeuralNetwork:
    """
    Simple Feed-forward Neural Network from scratch (NumPy only)
    
    Example usages:
    - Binary classification: hidden_layers=[64], output_activation='sigmoid', loss='bce'
    - Multi-class:         hidden_layers=[128, 64], output_activation='softmax', loss='ce'
    - Regression:          hidden_layers=[32], output_activation='linear', loss='mse'
    """
    def __init__(self,
                 input_size,
                 hidden_layers=[64, 32],           # list of hidden layer sizes
                 output_size=1,
                 output_activation='sigmoid',      # 'sigmoid', 'softmax', 'linear'
                 loss='bce',                       # 'mse', 'bce', 'ce'
                 learning_rate=0.01,
                 momentum=0.9,
                 random_state=42):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.lr = learning_rate
        self.momentum = momentum
        self.loss_name = loss.lower()
        self.output_act = output_activation.lower()

        np.random.seed(random_state)
        
        # Layer sizes: input → hidden1 → hidden2 → ... → output
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights & biases
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[i]
            # He initialization for ReLU, Xavier for sigmoid/softmax
            if self.output_act in ['sigmoid', 'softmax'] and i == len(self.layer_sizes)-2:
                std = np.sqrt(1.0 / fan_in)
            else:
                std = np.sqrt(2.0 / fan_in)  # He init
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * std
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Momentum velocity
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    def _forward(self, X):
        """Forward pass. Returns list of activations and pre-activations (z)"""
        activations = [X]           # a[0] = input
        zs = []                     # pre-activations z = W*a + b
        
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            zs.append(z)
            
            if i < len(self.weights)-1:
                a = relu(z)             # hidden layers → ReLU
            else:
                # Output layer activation
                if self.output_act == 'sigmoid':
                    a = sigmoid(z)
                elif self.output_act == 'softmax':
                    a = softmax(z)
                else:  # linear (regression)
                    a = z
            activations.append(a)
            
        return activations, zs

    def _loss(self, y_true, y_pred):
        if self.loss_name == 'mse':
            return np.mean((y_true - y_pred)**2)
        elif self.loss_name == 'bce':
            y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        elif self.loss_name == 'ce':
            y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return 0.0

    def _loss_deriv(self, y_true, y_pred, last_z):
        if self.loss_name == 'mse':
            return (y_pred - y_true) * 2 / y_true.shape[0]
        elif self.loss_name == 'bce':
            return (y_pred - y_true) / y_true.shape[0]
        elif self.loss_name == 'ce':
            return (y_pred - y_true) / y_true.shape[0]
        return y_pred - y_true

    def fit(self, X, y, epochs=1000, batch_size=32, verbose=1):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuf = X[indices]
            y_shuf = y[indices]
            
            total_loss = 0
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]
                
                # Forward
                activations, zs = self._forward(X_batch)
                y_pred = activations[-1]
                
                # Loss
                batch_loss = self._loss(y_batch, y_pred)
                total_loss += batch_loss * (end - start)
                
                # Backward
                delta = self._loss_deriv(y_batch, y_pred, zs[-1])
                
                # Output layer activation derivative
                if self.output_act == 'sigmoid':
                    delta *= sigmoid_deriv(zs[-1])
                elif self.output_act == 'softmax':
                    # For CE + softmax, derivative is already simplified (y_pred - y_true)
                    pass
                # linear → no extra multiply
                
                deltas = [delta]
                
                # Backprop through hidden layers
                for i in range(len(self.weights)-1, 0, -1):
                    delta = deltas[0] @ self.weights[i].T * relu_deriv(zs[i-1])
                    deltas.insert(0, delta)
                
                # Update weights & biases with momentum
                for i in range(len(self.weights)):
                    grad_w = activations[i].T @ deltas[i]
                    grad_b = np.sum(deltas[i], axis=0, keepdims=True)
                    
                    self.v_w[i] = self.momentum * self.v_w[i] - self.lr * grad_w
                    self.v_b[i] = self.momentum * self.v_b[i] - self.lr * grad_b
                    
                    self.weights[i] += self.v_w[i]
                    self.biases[i] += self.v_b[i]
            
            avg_loss = total_loss / n_samples
            if verbose and (epoch+1) % max(1, epochs//20) == 0:
                print(f"Epoch {epoch+1}/{epochs}   loss = {avg_loss:.6f}")

        return self

    def predict(self, X):
        activations, _ = self._forward(X)
        return activations[-1]

    def predict_proba(self, X):
        p = self.predict(X)
        if self.output_act == 'sigmoid':
            return np.hstack([1-p, p])
        elif self.output_act == 'softmax':
            return p
        return p


# ────────────────────────────────────────────────
#                   Quick Examples
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # ── Binary classification ───────────────────────────────
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=12,
                               random_state=42)
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nn_clf = SimpleNeuralNetwork(
        input_size=20,
        hidden_layers=[32, 16],
        output_size=1,
        output_activation='sigmoid',
        loss='bce',
        learning_rate=0.03,
        momentum=0.9
    )
    nn_clf.fit(X_train, y_train, epochs=400, batch_size=64, verbose=1)

    proba = nn_clf.predict(X_test)
    pred = (proba >= 0.5).astype(int).ravel()
    print("\nBinary classification accuracy:", accuracy_score(y_test, pred))

    # ── Regression example ──────────────────────────────────
    X_reg, y_reg = make_regression(n_samples=1200, n_features=15, noise=20, random_state=42)
    X_reg = scaler.fit_transform(X_reg)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    nn_reg = SimpleNeuralNetwork(
        input_size=15,
        hidden_layers=[64, 32],
        output_size=1,
        output_activation='linear',
        loss='mse',
        learning_rate=0.005,
        momentum=0.85
    )
    nn_reg.fit(X_tr_r, y_tr_r, epochs=600, batch_size=32)

    y_pred_r = nn_reg.predict(X_te_r).ravel()
    print("Regression RMSE:", np.sqrt(mean_squared_error(y_te_r, y_pred_r)))