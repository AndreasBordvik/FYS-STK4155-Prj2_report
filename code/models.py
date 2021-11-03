import numpy as np
from common import MSE
from sklearn.preprocessing import StandardScaler

# Activation functions


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01*x)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def binary_classifier(x):
    return np.where(x >= 0, 1, 0)


class Fixed_layer:
    def __init__(self, nbf_inputs: int, nbf_outputs: int, weights, bias, activation_function=lambda x: x):
        self.input = nbf_inputs
        self.output = nbf_outputs
        self.activation = activation_function
        # np.random.rand(nbf_inputs, nbf_outputs) # TODO: include possible negative weight initialization
        self.weights = weights
        # np.random.rand(1, nbf_outputs) # TODO: include possible negative weight initialization
        self.bias = bias
        self.accumulated_gradient = np.zeros_like(self.weights)

    def forward_prop(self, input_: np.ndarray) -> np.ndarray:
        z = (input_ @ self.weights) + self.bias
        a = self.activation(z)
        return a, self.weights, self.bias


class Layer:
    def __init__(self, nbf_inputs: int, nbf_outputs: int, activation_function=lambda x: x, name="name"):
        self.name = name
        self.input = nbf_inputs
        self.output = nbf_outputs
        self.activation = lambda z: np.exp(z) / (np.exp(z) + 1)
        self.grad_activation = lambda a: a*(1-a)
        # TODO: include possible negative weight initialization
        self.weights = np.random.randn(
            nbf_inputs, nbf_outputs)  # TODO:Must be normal
        # TODO: include possible negative weight initialization
        self.bias = np.random.randn(1, nbf_outputs)
        # self.accumulated_gradient = np.zeros_like(self.weights)
        self.z = None
        self.a = None
        self.deltas = None
        # self.grad_activation = grad(activation_function)

    def forward_prop(self, input_: np.ndarray) -> np.ndarray:
        self.z = (input_ @ self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a, self.weights, self.bias


class NeuralNetwork:
    def __init__(self, cost=MSE, learning_rate=0.001):
        self.sequential_layers = []
        self.cost = cost
        self.grad_cost = None  # grad(cost)
        self.eta = learning_rate

    def add_layer(self, layer: Layer):
        self.sequential_layers.append(layer)

    def predict(self, input_):
        X = input_.copy()
        for layer in self.sequential_layers:
            #print(X, '\n')
            X, w, b = layer.forward_prop(X)
            #print(w, '\n')
            #print(b, '\n')

        return X

    def fit(self, X, t):  # fit using feed forward and backprop
        _ = self.predict(X)  # t_hat = output activation

        # Backprop
        # Calculating the gradient of the error at the output
        output_layer = self.sequential_layers[-1]
        a_out = output_layer.a  # output activation
        print("a_out.shape:", a_out.shape)
        print("t.shape:", t.reshape(-1, 1).shape)
        print("output_layer.grad_activation(a_out).shape:",
              output_layer.grad_activation(a_out).shape)

        output_layer.deltas = output_layer.grad_activation(
            a_out) * (t.reshape(-1, 1) - a_out)

        for i in range(len(self.sequential_layers)-1, 0, -1):

            current = self.sequential_layers[i-1]
            print("current:", current.name)
            right = self.sequential_layers[i]
            print("right:", right.name)
            print("current.weights.T.shape", current.weights.T.shape)
            print("right.deltas.T.shape", right.deltas.T.shape)
            print("right.deltas.shape", right.deltas.shape)
            a_cur = current.a
            current.deltas = current.grad_activation(
                a_cur) * (right.deltas @ right.weights.T)
            #current.deltas = 1

        print("backprop is done")
        print("Updating the weights")

        for i in range(len(self.sequential_layers)-1, 0, -1):
            print("weights before:")
            print(current.weights)
            current = self.sequential_layers[i-1]
            right = self.sequential_layers[i]

            print("right.deltas.shape:", right.deltas.shape)

            print("current.a.shape:", current.a.shape)
            current.weights -= self.eta * right.deltas * current.a
            print("weights updated:")
            print(current.weights)

        """
            for i in range(len(self.sequential_layers)-2, 0, -1):
            current = self.sequential_layers[i]
            right = self.sequential_layers[i+1]

            for j in range(num_samples):  # X.shape[0]?
                current.weights -= eta*current.deltas[j]*right.activated[j]"""

        # error = self.grad_cost(y, y_hat)
        # self.sequential_layers[-1].deltas = error
        # for layer in reversed(self.sequential_layers):
        #     error = layer.backprop(error)
        #     layer.delta = error


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
