# import autograd.numpy as np
import numpy as np
from common import MSE
from sklearn.preprocessing import StandardScaler
from autograd import elementwise_grad
import tensorflow as tf
from sklearn.neural_network import MLPRegressor


# Activation functions
def relu(x):
    return np.maximum(0, x)


def grad_relu(x):
    return np.greater(x, 0).astype(int)
    return np.where(x <= 0, 0, 1).astype(int)


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
    def __init__(self, nbf_inputs: int, nbf_neurons: int, activation="sigmoid", name="name"):

        pick_activation = {"sigmoid": [
            sigmoid, grad_sigmoid], "relu": [relu, grad_relu]}

        self.name = name
        self.input = nbf_inputs
        self.neurons = nbf_neurons
        self.activation = pick_activation[activation][0]
        self.grad_activation = pick_activation[activation][1]
        self.weights = np.random.randn(nbf_inputs, nbf_neurons)
        # TODO: include possible negative weight initialization
        self.bias = np.zeros(nbf_neurons) + 0.01
        self.z = None
        self.output = None
        self.error = None
        self.deltas = 0  # The gradient of the error

    def forward_prop(self, input_: np.ndarray) -> np.ndarray:
        self.z = (input_ @ self.weights) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def __str__(self):
        return f"Layer name: {self.name}"


class NeuralNetwork:
    def __init__(self, cost=MSE, learning_rate=0.001, lmb=0, network_type="regression"):
        self.sequential_layers = []
        self.cost = cost
        self.grad_cost = None  # grad(cost)
        self.eta = learning_rate
        self.lmb = lmb
        self.network_type = network_type

    def add(self, layer: Layer):
        self.sequential_layers.append(layer)

    def predict(self, input_, threshold=0.5):
        X = input_.copy()
        for layer in self.sequential_layers:
            X = layer.forward_prop(X)

        if self.network_type == "classification":
            X = np.where(X > threshold, 1, 0)
        return X

    def fit(self, X, t, batch_size, epochs, verbose=False):
        # TODO: mention that our implementation is SGD with replacement
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
        t_hat = self.predict(X)  # t_hat = output activation
        output_layer = self.sequential_layers[-1]
        n = X.shape[0]
        # calulating the error at the output
        # nb... 1/n er foreksjelelig fra mini batch og GD
        # target-output...

        if self.network_type == "classification":
            output_layer.error = -(t.reshape(-1, 1) - t_hat)
        else:
            # (2/n) = SGD.. GD =
            output_layer.error = (1/n) * -2 * \
                (t.reshape(-1, 1) - output_layer.output)

        output_layer.error = output_layer.error + 2*self.lmb * output_layer.output

        # output_layer.error = (2/n)*(output_layer.output - t.reshape(-1, 1))  # (2/n) = SGD.. GD =
        # deriverer mtp output.

        # Calculating the gradient of the error from the output error
        output_layer.deltas = output_layer.error * \
            output_layer.grad_activation(output_layer.z)

        # All other layers
        for i in range(len(self.sequential_layers)-1, 0, -1):
            current = self.sequential_layers[i-1]
            right = self.sequential_layers[i]

            # calulating the error at the output
            current.error = right.deltas @ right.weights.T

            # Calculating the gradient of the error from the output error
            current.deltas = current.error * current.grad_activation(current.z)

        # updating weights
        for i in range(len(self.sequential_layers)-1, 0, -1):
            current = self.sequential_layers[i-1]
            right = self.sequential_layers[i]

            # updating weights
            right.weights = right.weights - self.eta * \
                (current.output.T @ right.deltas)

            # updating bias
            right.bias = right.bias - self.eta * np.sum(right.deltas, axis=0)

        # Updating weights and bias for first hidden layer
        first_hidden = self.sequential_layers[0]
        first_hidden.weights = first_hidden.weights - \
            self.eta * (X.T @ first_hidden.deltas)
        first_hidden.bias = first_hidden.bias - \
            self.eta * np.sum(first_hidden.deltas, axis=0)

        # clean deltas in layers
        for i in range(len(self.sequential_layers)):
            self.sequential_layers[i].deltas = 0.0
            self.sequential_layers[i].error = 0.0


def NN_regression_comparison(eta, nbf_features, batch_size, epochs, lmb=0,  hidden_size=50,  act_func="relu"):
    loss = "mse"
    # Tensorflow model
    tf_model = tf.keras.Sequential()
    tf_model.add(tf.keras.layers.Input(shape=(nbf_features,), name="input"))
    tf_model.add(tf.keras.layers.Dense(hidden_size, activation=act_func,
                 kernel_regularizer=tf.keras.regularizers.L2(lmb), name="hidden1"))
    tf_model.add(tf.keras.layers.Dense(1, name="output"))
    tf_model.compile(loss=loss, optimizer=tf.optimizers.SGD(learning_rate=eta))

    # SKlearn model

    sk_model = MLPRegressor(hidden_layer_sizes=(hidden_size, ), solver='sgd', max_iter=epochs,
                            alpha=lmb, activation="logistic" if act_func is "sigmoid" else act_func,
                            learning_rate_init=eta, batch_size=batch_size)

    # Own implemented NN model
    NN_model = NeuralNetwork(cost=MSE, learning_rate=eta,
                             lmb=lmb, network_type="regression")
    NN_model.add(Layer(nbf_features, hidden_size,
                 activation=act_func, name="hidden1"))
    NN_model.add(Layer(hidden_size, 1, name="output", activation=act_func))

    return NN_model, sk_model, tf_model


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
