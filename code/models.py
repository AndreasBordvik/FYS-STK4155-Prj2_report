import numpy as np


class own_LinRegGD():
    def __init__(self):
        self.f = lambda X,W: X @ W
        self.scaled_data = False
        self.scaled_type = "normalized"
    
    def fit(self, 
            X_train, 
            t_train, 
            gamma = 0.1, 
            epochs = 10, 
            diff = 0.001, 
            scaledata=False,
            scaletype="normalized"):
        
        self.scaled_data = scaledata
        self.scaled_type = scaletype
        if(scaledata):
            X_train = self.scaler(X_train, scaletype) #Scaling before adding bias
        (k, m) = X_train.shape
        X_train = self.add_bias(X_train)
        self.theta = theta = np.zeros(m+1)
        nbOfEpochs = 0 
        
        for i in range(epochs):
            nbOfEpochs += 1
            update = 2/k * gamma *  X_train.T @ (self.f(X_train,self.theta) - t_train)
            self.theta -= update
            if(abs(update) < diff).all():
                return nbOfEpochs+1
        return i+1
           
    def predict(self, x, threshold=0.5):
        if(self.scaled_data):
            x = self.scaler(x, self.scaled_type) #Scaling before adding bias
        z = self.add_bias(x)
        y_pred = z @ self.theta
        return y_pred
    
    def add_bias(self, x):
        # Bias element = 1 is inserted at index 0
        return np.insert(x, 0, 1, axis=1)




if __name__ == '__main__':
    print("Import this file as a package please!")
