import numpy as np 


class LogisticRegression: 
    def __init__(self,lr=0.001,n_iterations=1000) : 
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights=None
        self.bias =None
        self.cost_history=[]

    def sigmoid(self,z) : 
        return 1/(1+np.exp(-z))
    
    def cost(self, h, y):
        """Cross-entropy loss"""
        m = len(y)
        return - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def fit(self,X,y) : 
        m,n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        for _ in range(self.n_iterations) : 
            z = np.dot(X,self.weights)+self.bias
            h=self.sigmoid(z) 
            dw=(1/m)*(np.dot(X.T,h-y))
            db=(1/m)*(np.sum(h-y))

            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            self.cost_history.append(self.cost(h,y))
        
    def predict(self,X) : 
        return (self.sigmoid(np.dot(X,self.weights)+self.bias)>=0.5).astype(int)