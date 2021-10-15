import numpy as np
from sklearn.preprocessing import StandardScaler


class own_LinRegGD():
    def __init__(self):
        self.f = lambda X,W: X @ W

    def fit(self, X_train, t_train, gamma = 0.1, epochs = 10, diff = 0.001):
        k, m = X_train.shape
        #X_train = self.add_bias(X_train)
        self.theta = np.zeros((m,1))
        trained_epochs = 0 
        
        for i in range(epochs):
            trained_epochs += 1
            update = 2/k * gamma *  X_train.T @ (self.f(X_train,self.theta) - t_train)
            self.theta -= update
            if(abs(update) < diff).all():
                print(f"Training stops at epoch: {trained_epochs}. Convergence - weights are updated less than diff {diff}")
                return trained_epochs
        return trained_epochs
        
        
           
    def predict(self, X):
        #z = self.add_bias(X)
        t_pred = X @ self.theta
        return t_pred
    
    def add_bias(self, x):
        # Bias element = 1 is inserted at index 0
        return np.insert(x, 0, 1, axis=1)


if __name__ == '__main__':
    print("Import this file as a package please!")
