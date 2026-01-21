import numpy as np 


class LinearRegression : 
    def __init__(self,lr=0.001,n_iterations = 1000,penalty=None): 
        self.lr=lr
        self.n_iterations = n_iterations 
        self.weights = None
        self.bias = None
        self.penalty=penalty

    def fit(self,X,y) : 
        m,n=X.shape
        self.weights=np.zeros(n)
        self.bias=0
        prev_loss = 0
        tol = 1e-5
        for _ in range(self.n_iterations) : 
            y_pred = self.predict(X) 
            dw=(1/m)*(np.dot(X.T,y_pred-y))
            if self.penalty=="l1":
                dw += self.lr*np.sign(self.weights)
            elif self.penalty =="l2" : 
                dw += self.lr*2*self.weights
            db=(1/m)*(np.sum(y_pred-y))

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

            current_loss = np.mean(np.sqrt(y_pred-y))
            if prev_loss-current_loss<tol:
                break
            prev_loss = current_loss

    def predict(self,X) : 
        return np.dot(X,self.weights)+self.bias 
         