import numpy as np
from sklearn.preprocessing import StandardScaler

# Activation functions
def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def binary_classifier(x):
    return np.where(x>=0,1,0)

class Fixed_layer:
    def __init__(self, nbf_inputs : int, nbf_outputs : int, weights, bias, activation_function=lambda x: x):
        self.input = nbf_inputs
        self.output = nbf_outputs
        self.activation = activation_function
        self.weights = weights#np.random.rand(nbf_inputs, nbf_outputs) # TODO: include possible negative weight initialization
        self.bias = bias#np.random.rand(1, nbf_outputs) # TODO: include possible negative weight initialization
        self.accumulated_gradient = np.zeros_like(self.weights)

    def forward_prop(self, input_ : np.ndarray) -> np.ndarray:
        z = (input_ @ self.weights) + self.bias
        a = self.activation(z)
        return a, self.weights, self.bias

class Layer:
    def __init__(self, nbf_inputs : int, nbf_outputs : int, activation_function=lambda x: x):
        self.input = nbf_inputs
        self.output = nbf_outputs
        self.activation = activation_function
        self.weights = np.random.rand(nbf_inputs, nbf_outputs) -0.5 # TODO: include possible negative weight initialization
        self.bias = np.random.rand(1, nbf_outputs) -0.5# TODO: include possible negative weight initialization
        self.accumulated_gradient = np.zeros_like(self.weights)

    def forward_prop(self, input_ : np.ndarray) -> np.ndarray:
        z = (input_ @ self.weights) + self.bias
        a = self.activation(z)
        return a, self.weights, self.bias

    def backward_prop(self, X, y):
        h = self.forward_prop(X)
        error = y - y_hat 

        # TODO: DO backprop

class NeuralNetwork:
    def __init__(self):
        self.sequential_layers = []
        

    def add_layer(self, layer: Layer):
        self.sequential_layers.append(layer)

    def predict(self, input_):
        X = input_.copy()
        for layer in self.sequential_layers:
            print(X, '\n')
            X, w, b = layer.forward_prop(X)
            print(w, '\n')
            print(b, '\n')



        return X

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
