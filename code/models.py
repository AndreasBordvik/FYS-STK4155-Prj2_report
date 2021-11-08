import autograd.numpy as np
from common import MSE
from sklearn.preprocessing import StandardScaler
from autograd import elementwise_grad

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
    def __init__(self, nbf_inputs: int, nbf_outputs: int, activation=lambda x: x, name="name"):
        self.name = name
        self.input = nbf_inputs
        self.output = nbf_outputs
        #self.activation = lambda z: np.exp(z) / (np.exp(z) + 1)
        self.activation = activation
        self.grad_activation = elementwise_grad(self.activation)
        # self.grad_activation = lambda a: a*(1-a)
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
            #print(X, '\n')
            X, w, b = layer.forward_prop(X)
            #print(w, '\n')
            #print(b, '\n')

        return X

    def train_model(self, X, t, batch_size, epochs, verbose=False):
        n_batches = int(X.shape[0] // batch_size)
        Xt = np.concatenate((X, t), axis=1)
        
        for epoch in range(epochs):
            if verbose:
                print(f'Training epoch {epoch}/{epochs}')
            batches = np.take(Xt, np.random.permutation(Xt.shape[0]), axis=0)
            batches = np.array_split(batches, n_batches, axis=0)
            
            for i, batch in enumerate(batches):
                if verbose:
                    print(f'Epoch={epoch} | {(i + 1) / len(batches) * 100:.2f}%')
                xi = batch[:, :-1]
                yi = batch[:, -1].reshape(-1,1)
                self.fit(xi, yi)
            

    def fit(self, X, t):  # fit using feed forward and backprop
        _ = self.predict(X)  # t_hat = output activation
        # Backprop
        # Calculating the gradient of the error at the output
        output_layer = self.sequential_layers[-1]
        a_out = output_layer.a  # output activation
        output_layer.deltas = output_layer.grad_activation(a_out) * (t.reshape(-1, 1) - a_out)
        output_layer.deltas = np.mean(
            output_layer.deltas, axis=0, keepdims=True)

        output_layer.a = np.mean(output_layer.a, axis=0, keepdims=True)
        for i in range(len(self.sequential_layers)-1, 0, -1):
            current = self.sequential_layers[i-1]
            right = self.sequential_layers[i]
            a_cur = current.a
            current.deltas = current.grad_activation(
                a_cur) * (right.deltas @ right.weights.T)
            current.deltas = np.mean(current.deltas, axis=0, keepdims=True)
            current.a = np.mean(current.a, axis=0, keepdims=True)

        for i in range(len(self.sequential_layers)-1, 0, -1):
            current = self.sequential_layers[i-1]
            right = self.sequential_layers[i]
            right.weights -= self.eta * right.deltas * current.a.T

        first_hidden = self.sequential_layers[0]
        first_hidden.weights -= self.eta * first_hidden.deltas

        # clean deltas in layers 
        for i in range(len(self.sequential_layers)):
            self.sequential_layers[i].deltas = 0 
      

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
