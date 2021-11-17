import autograd.numpy as np
# Replacing numpy with autograd might break something, beware!
from autograd import grad
# import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from tqdm import tqdm
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from common import INPUT_DATA, REPORT_DATA, REPORT_FIGURES, EX_A, EX_B, EX_C, EX_D, EX_E, EX_F
from cmcrameri import cm
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data_utils


def cost_MSE(X, y, theta, lmb=0):
    return ((y - X @ theta)**2).sum() + lmb*(theta**2).sum()

# Activation functions


def return_self(x):
    return x


def grad_return_self(x):
    return 1


def relu(x):
    return np.maximum(0, x)


def grad_relu(x):
    return np.greater(x, 0).astype(int)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01*x)


def grad_leaky_relu(x):
    return np.where(x > 0, 1., 0.01)


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def grad_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def binary_classifier(x):
    return np.where(x >= 0, 1, 0)

# Learning rate schedulers


def lr_invscaling(eta, t, power_t=0.25):
    return eta / np.power(t, power_t)


def lr_expdecay(eta, t):
    return eta * np.exp(-0.1*t)

# Different SGD implementations


def sgd(X_train, t_train, theta, n_epoch, batch_size, eta, lr_scheduler=False, scheduler=lr_invscaling, lmb=0, d_cost_MSE=grad(cost_MSE, 2)):
    n_batches = int(X_train.shape[0] / batch_size)

    if lr_scheduler:
        eta0 = eta

    for epoch in tqdm(range(n_epoch), f"Training {n_epoch} epochs"):
        for i in range(n_batches):
            random_idx = batch_size*np.random.randint(n_batches)
            xi = X_train[random_idx:random_idx+batch_size]
            yi = t_train[random_idx:random_idx+batch_size]

            gradient = (2./batch_size)*d_cost_MSE(xi, yi, theta, lmb)

            if lr_scheduler:
                if epoch >= 10:  # Keep the initial learningrate for the first 10 samples
                    eta = scheduler(eta0, epoch*batch_size+(i+1))

            theta = theta - eta*gradient

    return theta.ravel()


def momentum_sgd(X_train, t_train, theta, n_epoch, batch_size, eta, beta=0.9, lr_scheduler=False, lmb=0, d_cost_MSE=grad(cost_MSE, 2)):
    n_batches = int(X_train.shape[0] / batch_size)

    if lr_scheduler:
        eta0 = eta

    for epoch in tqdm(range(n_epoch), f"Training {n_epoch} epochs"):
        momentum = 0

        for i in range(n_batches):
            random_idx = batch_size*np.random.randint(n_batches)
            xi = X_train[random_idx:random_idx+batch_size]
            yi = t_train[random_idx:random_idx+batch_size]

            gradient = (2./batch_size)*d_cost_MSE(xi, yi, theta, lmb)

            if lr_scheduler:
                eta = lr_invscaling(eta0, epoch*batch_size+(i+1))

            momentum = momentum*beta - eta*gradient
            theta = theta + momentum

    return theta.ravel()


def rmsprop(X_train, t_train, theta, n_epoch, batch_size, eta, beta=0.9, eps=10**(-8), lr_scheduler=False, lmb=0, d_cost_MSE=grad(cost_MSE, 2)):
    n_batches = int(X_train.shape[0] // batch_size)

    if lr_scheduler:
        eta0 = eta

    for epoch in tqdm(range(n_epoch), f"Training {n_epoch} epochs"):
        s = np.zeros((X_train.shape[-1], 1))

        for i in range(n_batches):
            random_idx = batch_size*np.random.randint(n_batches)
            xi = X_train[random_idx:random_idx+batch_size]
            yi = t_train[random_idx:random_idx+batch_size]

            gradient = (2./batch_size)*d_cost_MSE(xi, yi, theta, lmb)

            if lr_scheduler:
                eta = lr_invscaling(eta0, epoch*batch_size+(i+1))

            s = s*beta + (1 - beta)*np.power(gradient, 2)
            theta = theta - eta*(gradient/np.sqrt(s + eps))

    return theta.ravel()


def adam(X_train, t_train, theta, n_epoch, batch_size, eta, beta1=0.9, beta2=0.99, eps=10**(-8), lr_scheduler=False, lmb=0, d_cost_MSE=grad(cost_MSE, 2)):
    n_batches = int(X_train.shape[0] // batch_size)

    if lr_scheduler:
        eta0 = eta

    for epoch in tqdm(range(n_epoch), f"Training {n_epoch} epochs"):
        s = np.zeros((X_train.shape[-1], 1))
        m = np.zeros_like(s)

        for i in range(n_batches):
            random_idx = batch_size*np.random.randint(n_batches)
            xi = X_train[random_idx:random_idx+batch_size]
            yi = t_train[random_idx:random_idx+batch_size]

            gradient = (2./batch_size)*d_cost_MSE(xi, yi, theta, lmb)

            m = beta1*m + (1 - beta1)*gradient  # First moment
            s = beta2*s + (1 - beta2)*np.power(gradient, 2)  # Second moment

            m = m / (1 - np.power(beta1, i+1))
            s = s / (1 - np.power(beta2, i+1))

            if lr_scheduler:
                eta = lr_invscaling(eta0, epoch*batch_size+(i+1))

            theta = theta - eta * (m / (np.sqrt(s) + eps))

    return theta.ravel()

# Neural Network Code


class Fixed_layer:
    def __init__(self, nbf_inputs: int, nbf_outputs: int, weights, bias, activation="sigmoid", name="name"):
        pick_activation = {"sigmoid": [
            sigmoid, grad_sigmoid], "relu": [leaky_relu, grad_relu]}

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
    def __init__(self, nbf_inputs: int, nbf_neurons: int, activation="none", name="name"):

        pick_activation = {"sigmoid": [sigmoid, grad_sigmoid],
                           "relu": [relu, grad_relu],
                           "leaky_relu": [leaky_relu, grad_leaky_relu],
                           "none": [return_self, grad_return_self]}

        self.name = name
        self.input = nbf_inputs
        self.neurons = nbf_neurons
        self.activation = pick_activation[activation][0]
        self.grad_activation = pick_activation[activation][1]

        # Weight init strategies
        if activation == "sigmoid":
            # Xavier initializations (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
            # self.weights = np.random.randn(nbf_inputs, nbf_neurons)
            span = np.sqrt(6.0 / (nbf_inputs + nbf_neurons))
            self.weights = np.random.uniform(-span,
                                             span, size=(nbf_inputs, nbf_neurons))
        else:
            # He initializations (https://arxiv.org/pdf/1502.01852.pdf).
            self.weights = np.random.normal(
                size=(nbf_inputs, nbf_neurons)) * np.sqrt(2.0 / nbf_inputs)

        self.bias = np.zeros(nbf_neurons) + 0.01
        self.nbf_parameters = self.weights.size + self.bias.size
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
    def __init__(self, X_test, t_test,  learning_rate=0.001, lmb=0, network_type="regression"):
        self.sequential_layers = []
        self.grad_cost = None  # grad(cost)
        self.eta = learning_rate
        self.lmb = lmb
        self.network_type = network_type
        self.train_losses = []
        self.test_losses = []
        self.X_test = X_test
        self.t_test = t_test
        self.nbf_parameters = 0

    def add(self, layer: Layer):
        self.sequential_layers.append(layer)
        self.nbf_parameters += layer.nbf_parameters

    def predict(self, input_):
        X = input_.copy()
        for layer in self.sequential_layers:
            X = layer.forward_prop(X)

        return X

    def logistic_predict(self, input_, threshold=0.5):
        X = input_.copy()
        for layer in self.sequential_layers:
            X = layer.forward_prop(X)

        return np.where(X > threshold, 1, 0)

    def fit(self, X, t, batch_size, epochs, lr_scheduler=False, verbose=False):
        # TODO: mention that our implementation is SGD with replacement
        n_batches = int(X.shape[0] // batch_size)
        self.train_losses = []
        self.test_losses = []

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

                if lr_scheduler:
                    self.eta = lr_invscaling(self.eta, epoch*batch_size+(i+1))

            t_hat_train = self.predict(X.copy())
            self.train_losses.append(MSE(t.copy(), t_hat_train))

            t_hat_test = self.predict(self.X_test)
            self.test_losses.append(MSE(self.t_test, t_hat_test))

        return np.array(self.train_losses), np.array(self.test_losses)

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

    def __str__(self):
        return f"Number of network parameters: {self.nbf_parameters}"


def NN_regression_comparison(eta, nbf_features, batch_size, epochs, X_test, t_test, lmb=0,  hidden_size=50,  act_func="relu"):
    # Tensorflow model
    tf_model = tf.keras.Sequential()
    tf_model.add(tf.keras.layers.Input(shape=(nbf_features,), name="input"))
    tf_model.add(tf.keras.layers.Dense(hidden_size, activation=act_func,
                 kernel_regularizer=tf.keras.regularizers.L2(lmb), name="hidden1"))
    tf_model.add(tf.keras.layers.Dense(1, name="output"))
    tf_model.compile(
        loss="mse", optimizer=tf.optimizers.SGD(learning_rate=eta))

    # SKlearn model
    sk_model = MLPRegressor(hidden_layer_sizes=(hidden_size, ), solver='sgd', max_iter=epochs,
                            alpha=lmb, activation="logistic" if act_func == "sigmoid" else act_func,
                            learning_rate_init=eta, batch_size=batch_size)

    # Own implemented NN model
    NN_model = NeuralNetwork(cost=MSE, learning_rate=eta,
                             lmb=lmb, network_type="regression", X_test=X_test, t_test=t_test)
    NN_model.add(Layer(nbf_features, hidden_size,
                 activation=act_func, name="hidden1"))
    NN_model.add(Layer(hidden_size, 1, name="output", activation=act_func))

    return NN_model, sk_model, tf_model


def NN_simple_architecture(eta, nbf_features, problem_type, X_test, t_test, nbf_outputs=1, lmb=0,  hidden_size=25,  act_func="relu"):
    # Own implemented NN model
    NN_model = NeuralNetwork(learning_rate=eta, lmb=lmb,
                             network_type=problem_type, X_test=X_test, t_test=t_test)
    NN_model.add(Layer(nbf_features, hidden_size,
                 activation=act_func, name="hidden1"))
    NN_model.add(Layer(hidden_size, nbf_outputs,
                 name="output", activation=act_func))

    # Tensorflow model
    act_func = tf.keras.layers.LeakyReLU(
        alpha=0.01) if act_func == "leaky_relu" else act_func
    tf_model = tf.keras.Sequential()
    tf_model.add(tf.keras.layers.Input(shape=(nbf_features,), name="input"))
    tf_model.add(tf.keras.layers.Dense(hidden_size, activation=act_func,
                 kernel_regularizer=tf.keras.regularizers.L2(lmb), name="hidden1"))
    tf_model.add(tf.keras.layers.Dense(nbf_outputs, name="output"))

    if problem_type == "classification":
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        loss = tf.keras.losses.MeanSquaredError()
    tf_model.compile(
        loss=loss, optimizer=tf.optimizers.SGD(learning_rate=eta))

    return NN_model, tf_model


def NN_large_architecture(eta, nbf_features, problem_type, X_test, t_test, nbf_outputs=1, lmb=0,  hidden_size=25,  act_func="relu"):
    # Own implemented NN model
    NN_model = NeuralNetwork(learning_rate=eta, lmb=lmb,
                             network_type=problem_type, X_test=X_test, t_test=t_test)
    h1 = Layer(nbf_features, hidden_size,
               activation=act_func, name="hidden1")
    NN_model.add(h1)
    h2 = Layer(h1.neurons, hidden_size*2,
               activation=act_func, name="hidden2")
    NN_model.add(h2)
    h3 = Layer(h2.neurons, hidden_size, activation=act_func, name="hidden3")
    NN_model.add(h3)
    h4 = Layer(h3.neurons, np.ceil(hidden_size//3).astype(int),
               activation=act_func, name="hidden4")
    NN_model.add(h4)
    out = Layer(h4.neurons, nbf_outputs, name="output")
    NN_model.add(out)

    # Tensorflow model
    act_func = tf.keras.layers.LeakyReLU(
        alpha=0.01) if act_func == "leaky_relu" else act_func
    tf_model = tf.keras.Sequential()
    input_tf = tf.keras.layers.Input(shape=(nbf_features,), name="input")
    tf_model.add(input_tf)
    h1_tf = tf.keras.layers.Dense(hidden_size, activation=act_func,
                                  kernel_regularizer=tf.keras.regularizers.L2(lmb), name="hidden1")
    tf_model.add(h1_tf)
    h2_tf = tf.keras.layers.Dense(hidden_size*2, activation=act_func,
                                  kernel_regularizer=tf.keras.regularizers.L2(lmb), name="hidden2")
    tf_model.add(h2_tf)
    h3_tf = tf.keras.layers.Dense(hidden_size, activation=act_func,
                                  kernel_regularizer=tf.keras.regularizers.L2(lmb), name="hidden3")
    tf_model.add(h3_tf)
    h4_tf = tf.keras.layers.Dense(np.ceil(hidden_size//3).astype(int), activation=act_func,
                                  kernel_regularizer=tf.keras.regularizers.L2(lmb), name="hidden4")
    tf_model.add(h4_tf)
    out_tf = tf.keras.layers.Dense(nbf_outputs, name="output")
    tf_model.add(out_tf)

    if problem_type == "classification":
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        loss = tf.keras.losses.MeanSquaredError()
    tf_model.compile(loss=loss, optimizer=tf.optimizers.SGD(learning_rate=eta))

    return NN_model, tf_model

# cmap=cm.lajolla
# cmap="RdYlGn_r"


def plot_save_NN_results(parameters: int, model_size: str,
                         eta_list: np.ndarray, lmb_list: np.ndarray,
                         heatmap_mtrx: np.ndarray, heatmap_mtrx_tf: np.ndarray,
                         path: str,
                         activation_type: str):
    # Own NN heatmap
    plt.figure(figsize=(12, 10))
    eta_list = np.around(eta_list, decimals=6)
    lmb_list = np.around(lmb_list, decimals=6)
    gridsearch = sns.heatmap(heatmap_mtrx, annot=True, fmt=".4f",
                             xticklabels=eta_list, yticklabels=lmb_list, cmap=cm.lajolla)
    gridsearch.invert_xaxis()
    gridsearch.invert_yaxis()
    gridsearch.set_xticklabels(gridsearch.get_xticklabels(), rotation=80)

    plt.title(
        f"Own NN implementation - {model_size} architecture\nEta and Lambda gridsearch on model having {parameters} parameters")
    plt.xlabel("Eta")
    plt.ylabel("Lambda")
    plt.savefig(
        f"{path}heatmap_own_NN_{model_size}_parameters_{parameters}_{activation_type}.pdf")
    # {REPORT_FIGURES}{EX_B}
    # Tensorflow heatmap
    plt.figure(figsize=(12, 10))
    gridsearch_tf = sns.heatmap(heatmap_mtrx_tf, annot=True,
                                xticklabels=eta_list, yticklabels=lmb_list, cmap=cm.lajolla)
    gridsearch_tf.invert_xaxis()
    gridsearch_tf.invert_yaxis()
    gridsearch_tf.set_xticklabels(gridsearch_tf.get_xticklabels(), rotation=80)

    plt.title(
        f"Tensorflow NN implementation - {model_size} architecture\nEta and Lambda gridsearch on model having {parameters} parameters")
    plt.xlabel("Eta")
    plt.ylabel("Lambda")
    plt.savefig(
        f"{path}heatmap_tf_{model_size}_parameters_{parameters}_{activation_type}.pdf")

# Linear Regression code


class own_LinRegGD():
    def __init__(self):
        self.f = lambda X, W: X @ W

    def fit(self, X_train, t_train, gamma=0.1, epochs=10, diff=0.001):
        k, m = X_train.shape
        # X_train = self.add_bias(X_train)
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
        # z = self.add_bias(X)
        t_pred = X @ self.theta
        return t_pred

    def add_bias(self, x):
        # Bias element = 1 is inserted at index 0
        return np.insert(x, 0, 1, axis=1)


if __name__ == '__main__':
    print("Import this file as a package please!")


class LogReg():
    def __init__(self, eta=None, lmb=None) -> None:
        super().__init__()
        self.eta = eta
        self.lmb = lmb

    def fit(self, X_train, t_train, batch_size, epochs, solver="SGD"):
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""
        m = X_train.shape[1]
        n_batches = np.ceil(X_train.shape[0] / batch_size)

        self.beta = np.zeros(m)

        indicies = np.arange(X_train.shape[0])
        if solver == "SGD":
            # Stochastic Gradient Descent:

            for epoch in range(epochs):
                np.random.shuffle(indicies)
                minibatch_list = np.array_split(indicies, n_batches)

                for minibatch in (minibatch_list):
                    xi = np.take(X_train, minibatch, axis=0)
                    yi = np.take(t_train, minibatch, axis=0)
                    p = self.forward(xi)
                    gradient = -xi.T @ (yi-p) + 2*self.lmb*self.beta

                    # beta punishing
                    # updating betas:
                    self.beta = self.beta - self.eta*gradient

        elif solver == "NRM":
            # Newton Raphson´s Method:
            for epoch in range(epochs):
                self.newton_raphson(X_train, t_train)

        else:
            print("Choose SGM og NRM as solver")

    def accuracy(self, X_test, y_test, **kwargs):
        pred = self.predict(X_test, **kwargs)
        if len(pred.shape) > 1:
            pred = pred[:, 0]
        return sum(pred == y_test) / len(pred)

    def forward(self, X):
        return 1/(1+np.exp(-X @ self.beta))

    def newton_raphson(self, X, y):
        p = self.forward(X)
        p_1 = 1-p
        score = -X.T @ (y-p)

        W = np.zeros(shape=(len(y), len(y)))
        np.fill_diagonal(W, (p*p_1))

        hessian = X.T@W@X
        update = np.linalg.pinv(hessian) @ score

        self.beta -= update

    def predict(self, x, threshold=0.5):
        # z = self.add_bias(x)
        score = self.forward(x)
        # score = z @ self.theta
        return (score > threshold).astype('int')


class TorchNeuralNetwork(nn.Module):
    def __init__(self, eta, lmb, input_dim, hidden_size):
        super(TorchNeuralNetwork, self).__init__()
        self.eta = eta
        self.lmb = lmb
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.linear_stack = nn.Sequential(

            nn.Linear(self.input_dim, self.hidden_size),  # hidden1
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2),  # hidden2
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),  # hidden3
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),  # output
            nn.Sigmoid()
        )
        self.optimizer = optim.SGD(
            self.parameters(), lr=self.eta, weight_decay=self.lmb)
        self.criterion = nn.BCELoss()

    def fit(self, epochs, trainloader):
        for epoch in range(epochs):
            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                # unsqueeze to broadcast:
                labels = labels.unsqueeze(-1)
                # zero gradients to avoid accum.grads
                self.optimizer.zero_grad()

                # forward -> backward -> optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

    def torch_accuracy(self, X_test, y_test):

        y_hat = self.forward(X_test)
        prediction_val = [1 if x > 0.5 else 0 for x in y_hat.data.numpy()]
        correct_val = (prediction_val == y_test.numpy()).sum()
        return correct_val/len(y_test)
