# import autograd.numpy as np
import numpy as np
from common import MSE
from sklearn.preprocessing import StandardScaler
from autograd import elementwise_grad

# Activation functions


def relu(x):
    return np.maximum(0, x)


def grad_relu(x):
    return 0 if x == 0 else 1


def leaky_relu(x):
    return np.where(x > 0, x, 0.01*x)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def grad_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def binary_classifier(x):
    return np.where(x >= 0, 1, 0)


class Fixed_layer:
    def __init__(self, nbf_inputs: int, nbf_outputs: int, weights, bias, activation="sigmoid", name="name"):
        pick_activation = {"sigmoid": [
            sigmoid, grad_sigmoid], "relu": [relu, grad_relu]}

        self.input = nbf_inputs
        self.output = nbf_outputs

        self.activation = pick_activation[activation][0]
        self.grad_activation = pick_activation[activation][1]

        # np.random.rand(nbf_inputs, nbf_outputs) # TODO: include possible negative weight initialization
        self.weights = weights
        # np.random.rand(1, nbf_outputs) # TODO: include possible negative weight initialization
        self.bias = bias
        self.accumulated_gradient = np.zeros_like(self.weights)
        self.deltas = 0

    def forward_prop(self, input_: np.ndarray) -> np.ndarray:
        self.z = (input_ @ self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a


class Layer:
    def __init__(self, nbf_inputs: int, nbf_outputs: int, activation="sigmoid", name="name"):

        pick_activation = {"sigmoid": [
            sigmoid, grad_sigmoid], "relu": [relu, grad_relu]}

        self.name = name
        self.input = nbf_inputs
        self.output = nbf_outputs
        self.activation = pick_activation[activation][0]
        self.grad_activation = pick_activation[activation][1]
        self.weights = np.random.randn(nbf_inputs, nbf_outputs)  # TODO: include possible negative weight initialization
        self.bias = np.zeros(nbf_outputs) + 0.01 # TODO: include possible negative weight initialization
        self.z = None
        self.output = None
        self.error = None 
        self.deltas = 0 # The gradient of the error

    def forward_prop(self, input_: np.ndarray) -> np.ndarray:
        self.z = (input_ @ self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def __str__(self):
        return f"Layer name: {self.name}"

class NeuralNetwork:
    def __init__(self, cost=MSE, learning_rate=0.001):
        self.sequential_layers = []
        self.cost = cost
        self.grad_cost = None  # grad(cost)
        self.eta = learning_rate

    def add(self, layer: Layer):
        self.sequential_layers.append(layer)

    def predict(self, input_):
        X = input_.copy()
        for layer in self.sequential_layers:
            X = layer.forward_prop(X)
        return X

    def fit(self, X, t, batch_size, epochs, verbose=False):
        n_batches = int(X.shape[0] // batch_size)
        for epoch in range(epochs):
            if verbose:
                print(f'Training epoch {epoch}/{epochs}')

            for i in range(n_batches):
                if verbose:
                    print(f'Epoch={epoch} | {(i + 1) / n_batches * 100:.2f}%')

                random_idx = batch_size*np.random.randint(n_batches)
                xi = X[random_idx:random_idx+batch_size]
                yi = t[random_idx:random_idx+batch_size]
                self.backpropagation(xi, yi)

    def backpropagation(self, X, t):  # fit using feed forward and backprop 
        _ = self.predict(X)  # t_hat = output activation        
        output_layer = self.sequential_layers[-1]        
        n = X.shape[0]
        # calulating the error at the output
        # nb... 1/n er foreksjelelig fra mini batch og GD
        #target-output... 
        output_layer.error = (2/n) * -(t.reshape(-1, 1) - output_layer.output)  # (2/n) = SGD.. GD =
        #output_layer.error = (2/n)*(output_layer.output - t.reshape(-1, 1))  # (2/n) = SGD.. GD =
        # deriverer mtp output.
        
        # Calculating the gradient of the error from the output error
        output_layer.deltas = output_layer.error * output_layer.grad_activation(output_layer.z)
                
        # All other layers
        for i in range(len(self.sequential_layers)-1, 0, -1):
            current = self.sequential_layers[i-1]
            right = self.sequential_layers[i]
            
            # calulating the error at the output
            current.error = right.deltas @ right.weights.T
            # current.error = np.dot(right.weights, right.deltas)

            # Calculating the gradient of the error from the output error
            current.deltas = current.error * current.grad_activation(current.z) 
             
        # updating weights
        for i in range(len(self.sequential_layers)-1, 0, -1):
            current = self.sequential_layers[i-1]
            right = self.sequential_layers[i]
            right.weights = right.weights - self.eta * (current.output.T @ right.deltas)    
            right.bias = right.bias - self.eta * np.sum(right.deltas, axis=0)
            
        # updating for first hidden layer
        first_hidden = self.sequential_layers[0]
        first_hidden.weights = first_hidden.weights - self.eta * (X.T @ first_hidden.deltas)
        first_hidden.bias = first_hidden.bias - self.eta * np.sum(first_hidden.deltas, axis=0)

        # clean deltas in layers
        for i in range(len(self.sequential_layers)):
            self.sequential_layers[i].deltas = 0.0
            self.sequential_layers[i].error = 0.0


class own_LinRegGD():
    def __init__(self):
        self.f = lambda X, W: X @ W

    def fit(self, X_train, t_train, gamma=0.1, epochs=10, diff=0.001):
        k, m = X_train.shape
        #X_train = self.add_bias(X_train)
        self.theta = np.zeros((m, 1))
        trained_epochs = 0

        for i in range(epochs):
            trained_epochs += 1
            update = 2/k * gamma * \
                X_train.T @ (self.f(X_train, self.theta) - t_train)
            self.theta -= update
            if(abs(update) < diff).all():
                print(
                    f"Training stops at epoch: {trained_epochs}. Convergence - weights are updated less than diff {diff}")
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
