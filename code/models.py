#import numpy as np
#from common import MSE
import jax.numpy as jnp
from jax import random
from jax import grad, vmap
from sklearn.preprocessing import StandardScaler

key = random.PRNGKey(1)
# Activation functions


def relu(x):
    return jnp.maximum(0, x)


def leaky_relu(x):
    return jnp.where(x > 0, x, 0.01*x)


def sigmoid(x):
    return jnp.exp(x) / (1 + jnp.exp(x))


def binary_classifier(x):
    return jnp.where(x >= 0, 1, 0)


class Fixed_layer:
    def __init__(self, nbf_inputs: int, nbf_outputs: int, weights, bias, activation_function=lambda x: x):
        self.input = nbf_inputs
        self.output = nbf_outputs
        self.activation = activation_function
        # np.random.rand(nbf_inputs, nbf_outputs) # TODO: include possible negative weight initialization
        self.weights = weights
        # np.random.rand(1, nbf_outputs) # TODO: include possible negative weight initialization
        self.bias = bias
        self.accumulated_gradient = jnp.zeros_like(self.weights)

    def forward_prop(self, input_: jnp.ndarray) -> jnp.ndarray:
        z = (input_ @ self.weights) + self.bias
        a = self.activation(z)
        return a, self.weights, self.bias


class Layer:
    def __init__(self, nbf_inputs: int, nbf_outputs: int, activation_function=lambda x: x):
        self.input = nbf_inputs
        self.output = nbf_outputs
        self.activation = activation_function
        self.grad_activation = grad(activation_function)
        # TODO: include possible negative weight initialization
        self.weights = random.normal(key,(
            nbf_inputs, nbf_outputs))  # TODO:Must be normal
        # TODO: include possible negative weight initialization
        self.bias = random.normal(key,(1, nbf_outputs))
        # self.accumulated_gradient = np.zeros_like(self.weights)
        self.unactivated = None
        self.activated = None
        self.deltas = None
        # self.grad_activation = grad(activation_function)

    def forward_prop(self, input_: jnp.ndarray) -> jnp.ndarray:
        self.unactivated = (input_ @ self.weights) + self.bias
        self.activated = self.activation(self.unactivated)
        return self.activated, self.weights, self.bias


class NeuralNetwork:
    def __init__(self, cost=None):
        self.sequential_layers = []
        self.cost = cost
        self.grad_cost = None  # grad(cost)

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

    def backward_prop(self, X, y):
        y_hat = self.predict(X)
        output_layer = self.sequential_layers[-1]
        print("Computing errors for output layer")
        d_f_z = jnp.zeros(X.shape[0])
        for i in range(X.shape[0]):
            d_f_z = d_f_z.at[i].set(output_layer.grad_activation(output_layer.unactivated[i][0]))
        print(f"d_f_x: {d_f_z}")
        d_c_a = self.grad_cost(y, y_hat)
        output_layer.deltas = d_f_z * d_c_a

        for i in range(len(self.sequential_layers)-2, 0, -1):
            print(f"Computing errors for hidden layer #{i}")
            current = self.sequential_layers[i]
            right = self.sequential_layers[i+1]
            current.deltas = current.grad_activation(
                current.unactivated) * (current.weights @ right.deltas)

        for i in range(len(self.sequential_layers)-2, 0, -1):
            current = self.sequential_layers[i]
            right = self.sequential_layers[i+1]

            #for j in range(num_samples):  # X.shape[0]?
                #current.weights -= eta*current.deltas[j]*right.activated[j]

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
        self.theta = jnp.zeros((m, 1))
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
        return jnp.insert(x, 0, 1, axis=1)


if __name__ == '__main__':
    print("Import this file as a package please!")
