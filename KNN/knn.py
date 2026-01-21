import numpy as np 
from collections import Counter
def euclidean_distance(x,y) : 
    return np.sqrt(np.sum((x-y)**2))
class KNN : 
    def __init__(self,k=3) : 
        self.k = k 

    def fit(self,X,y) : 
        self.X_train = X
        self.y_train = y 

    def predict(self,X) : 
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    def _predict(self,x):
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        indices_points =np.argsort(distances)[:self.k]
        close_points = [self.y_train[i] for i in indices_points]
        most_common_label = Counter(close_points).most_common(1)
        return most_common_label[0][0]