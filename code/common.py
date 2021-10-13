import time
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
from tqdm import tqdm
from numpy.core.defchararray import index
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.linear_model as lm
from sklearn.utils import resample
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from sklearn.model_selection import KFold
from sklearn import linear_model
from typing import Callable, Tuple


# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/"  # Path for data ment for the report
REPORT_FIGURES = "../figures/"  # Path for figures ment for the report
#SEED_VALUE = 4155
EX1 = "EX1_"
EX2 = "EX2_"
EX3 = "EX3_"
EX4 = "EX4_"
EX5 = "EX5_"
EX6 = "EX6_"
EX6_1 = f"{EX6}{EX1}"
EX6_2 = f"{EX6}{EX2}"
EX6_3 = f"{EX6}{EX3}"
EX6_4 = f"{EX6}{EX4}"
EX6_5 = f"{EX6}{EX5}"


class Regression():
    """ Super class containing methods for fitting, predicting and producing stats for regression models.   
    """

    def __init__(self):
        self.betas = None
        self.X_train = None
        self.t_train = None
        self.t_hat_train = None
        self.param = None
        self.param_name = None
        self.SVDfit = None
        self.SE_betas = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Polymorph 

        """
        pass

    @property
    def get_all_betas(self) -> np.ndarray:
        """Returns predictor values

        Returns:
            [np.ndarray]: betas
        """
        return self.betas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Performs a prediction:       

        Args:
            X (np.ndarray): input data

        Returns:
            np.ndarray: Predicted values
        """

        prediction = X @ self.betas
        return prediction

    @property
    def SE(self):
        """Returns the standard error

        Returns:
            [type]: [description]
        """
        var_hat = (1./self.X_train.shape[0]) * \
            np.sum((self.t_train - self.t_hat_train)**2)

        if self.SVDfit:
            invXTX_diag = np.diag(SVDinv(self.X_train.T @ self.X_train))
        else:
            invXTX_diag = np.diag(np.linalg.pinv(
                self.X_train.T @ self.X_train))
        return np.sqrt(var_hat * invXTX_diag)

    def summary(self) -> pd.DataFrame:
        """Produces a summary with coeffs,  STD, confidence intervals

        Returns:
            [pd.DataFrame]: dataframe with values. 
        """
        # Estimated standard error for the beta coefficients
        N, P = self.X_train.shape
        SE_betas = self.SE

        # Calculating 95% confidence intervall
        CI_lower_all_betas = self.betas - (1.96 * SE_betas)
        CI_upper_all_betas = self.betas + (1.96 * SE_betas)

        # Summary dataframe
        params = np.zeros(self.betas.shape[0])
        params.fill(self.param)

        coeffs_df = pd.DataFrame.from_dict({f"{self.param_name}": params,
                                            "coeff name": [rf"$\beta${i}" for i in range(0, self.betas.shape[0])],
                                            "coeff value": np.round(self.betas, decimals=4),
                                            "std error": np.round(SE_betas, decimals=4),
                                            "CI lower": np.round(CI_lower_all_betas, decimals=4),
                                            "CI upper": np.round(CI_upper_all_betas, decimals=4)},
                                           orient='index').T

        return coeffs_df


class OLS(Regression):
    """Class for ordinary least squares regression. 

    Args:
        Regression ([Class]): Class to inherit. 
    """

    def __init__(self, degree=1, param_name="degree"):
        """init.

        Args:
            degree (int, optional): [description]. Defaults to 1.
            param_name (str, optional): [description]. Defaults to "degree".
        """
        super().__init__()
        self.param = degree
        self.param_name = param_name

    def fit(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Function to fit model

        Args:
            X (np.ndarray): Input data
            t (np.ndarray): target data

        Returns:
            np.ndarray: Predicted values. 
        """
        #self.SVDfit = SVDfit
        #self.keep_intercept = keep_intercept
        # if keep_intercept == False:
        #    X = X[:, 1:]

        self.X_train = X
        self.t_train = t

        # if SVDfit:
        #    self.betas = SVDinv(X.T @ X) @ X.T @ t
        # else:
        self.betas = np.linalg.pinv(X.T @ X) @ X.T @ t
        self.t_hat_train = X @ self.betas
        # print("betas.shape in train before squeeze:",self.betas.shape)
        self.betas = np.squeeze(self.betas)
        # print("betas.shape in train after squeeze:",self.betas.shape)
        return self.t_hat_train


class LinearRegression(OLS):
    """Linear regression model

    Args:
        OLS (Class): Class to inherit. 
    """

    def __init__(self):
        super().__init__()


class RidgeRegression(Regression):
    """Class for Ridge Regression

    Args:
        Regression (Class): Class to inherit
    """

    def __init__(self, lambda_val=1, param_name="lambda"):
        """init.

        Args:
            lambda_val (int, optional): [description]. Defaults to 1.
            param_name (str, optional): [description]. Defaults to "lambda".
        """
        super().__init__()
        self.param = self.lam = lambda_val
        self.param_name = param_name

    def fit(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Function to fit model and predict on same values. 

        Args:
            X (np.ndarray): Input data
            t (np.ndarray): Target data

        Returns:
            np.ndarray: Predicted values. 
        """
        #self.SVDfit = SVDfit
        #self.keep_intercept = keep_intercept
        # if keep_intercept == False:
        #    X = X[:, 1:]
        self.X_train = X
        self.t_train = t
        Hessian = X.T @ X
        # beta punishing and preventing the singular matix
        Hessian += self.lam * np.eye(Hessian.shape[0])

        # if SVDfit:
        #    self.betas = SVDinv(Hessian) @ X.T @ t
        # else:
        self.betas = np.linalg.pinv(Hessian) @ X.T @ t
        self.t_hat_train = X @ self.betas
        # print(f"Betas.shape in Ridge before:{self.betas.shape}")
        self.betas = np.squeeze(self.betas)
        # print(f"Betas.shape in Ridge after:{self.betas.shape}")
        return self.t_hat_train


def design_matrix(x: np.ndarray, features: int) -> np.ndarray:
    """Produces a design matrix for testing purposes. 

    Args:
        x (np.ndarray): input data
        features (int): number of feats.

    Returns:
        np.ndarray: Design matrix
    """
    X = np.zeros((x.shape[0], features))
    x = x.flatten()
    for i in range(1, X.shape[1]+1):
        X[:, i-1] = x ** i
    return X


def prepare_data(X: np.ndarray, t: np.ndarray, random_state, test_size=0.2, shuffle=True, scale_X=False, scale_t=False, skip_intercept=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Function to prepare data. Has the ability to set test size, shuffle, scale both X and t, and skip intercept. 

    Args:
        X (np.ndarray): Input data
        t (np.ndarray): Target Data
        random_state ([type]): Seed value
        test_size (float, optional): Size of test data. Defaults to 0.2.
        shuffle (bool, optional): Shuffles the data before split. Defaults to True.
        scale_X (bool, optional): Scales x. Defaults to False.
        scale_t (bool, optional): Scales target data. Defaults to False.
        skip_intercept (bool, optional): Skips intercept. Defaults to True.

    Returns:
        Tuple: Arrays containing X_train, X_test, t_train, t_test
    """
    X_train, X_test, t_train, t_test = train_test_split(
        X, t, test_size=test_size, shuffle=shuffle, random_state=random_state)

    # Scale data
    if(scale_X):
        X_train, X_test = standard_scaling(X_train, X_test)

    if(scale_t):
        t_train, t_test = standard_scaling(t_train, t_test)

    if (skip_intercept):
        X_train = X_train[:, 1:]
        X_test = X_test[:, 1:]

    return X_train, X_test, t_train, t_test


def manual_scaling(data: np.ndarray) -> np.ndarray:
    """    Avoids the use of sklearn StandardScaler(), which also
    divides the scaled value by the standard deviation.
    This scaling is essentially just a zero centering

    Args:
        data (np.ndarray): Input data

    Returns:
        np.ndarray: Scaled data
    """
    return data - np.mean(data, axis=0)


def standard_scaling(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scales data using the StandarScaler from sklearn.preprocessing

    Args:
        train (np.ndarray): Training data
        test (np.ndarray): test data

    Returns:
        Tuple[np.ndarray, np.ndarray]: Scaled data
    """
    scaler = StandardScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled


def standard_scaling_single(data):
    """Scales data using the StandarScaler from sklearn.preprocessing. For scaling a single dataset.    

    Args:
        data ([type]): Input data set       

    Returns:
        [type]: Scaled data
    """
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler


def min_max_scaling(data):
    """Scales data using the MinMaxScaler from sklearn.preprocessing

    Args:
        data ([type]): input data

    Returns:
        [type]: Scaled data
    """
    scaler = MinMaxScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler


def create_X(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Function based on code from course website. Creates design matrix. 

    Args:
        x (np.ndarray): input data
        y (np.ndarray): input data
        n (int): Number of degrees

    Returns:
        np.ndarray: Design Matrix
    """
    if (len(x.shape)) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X
    # return X[:, 1:]


def remove_intercept(X):
    """Removes the intercept from design matrix

    Args:
        X ([type]): Design matrix

    Returns:
        [type]: Design matrix with intercept removed. 
    """
    return X[:, 1:]


def FrankeFunction(x: float, y: float) -> float:
    """Function from course website. 

    Args:
        x (float): Input data
        y (float): input data

    Returns:
        float: [description]
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def timer(func) -> float:
    """
    Simple timer that can be used as a decorator to time functions
    """
    def timer_inner(*args, **kwargs):
        t0: float = time.time()
        result = func(*args, **kwargs)
        t1: float = time.time()
        print(
            f"Elapsed time {1000*(t1 - t0):6.4f}ms in function {func.__name__}"
        )
        return result
    return timer_inner


def FrankeFunctionMeshgrid() -> np.ndarray:
    """Making meshgrid of datapoints and compute Franke's function

    Returns:
        np.ndarray: [description]
    """
    n = 5
    N = 1000
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    z = FrankeFunction(x, y)
    X = create_X(x, y, n=n)
    return X


def bias_error_var(t_true: np.ndarray, t_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns bias, variance and MSE 

    Args:
        t_true ([np.ndarray]): predicted data
        t_pred ([np.ndarray]): ground truth

    Returns:
        [Tuple]: bias, variance and MSE
    """
    error = np.mean(np.mean((t_true - t_pred)**2, axis=1, keepdims=True))
    bias = np.mean((t_true - np.mean(t_pred, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(t_pred, axis=1, keepdims=True))
    return error, bias, variance


def plot_franke_function() -> plt:
    """Plots the franke function for some set values.  

    Returns:
        plt: plot. 
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def SVDinv(A: np.ndarray) -> np.ndarray:
    """Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.

    Args:
        A ([type]): Matrix

    Returns:
        [type]: inverse of A
    """

    U, s, VT = np.linalg.svd(A)

    D = np.zeros((len(U), len(VT)))
    D = np.diag(s)
    # print(D)
    UT = np.transpose(U)
    V = np.transpose(VT)
    invD = np.linalg.inv(D)
    return V@(invD@UT)


@timer
def bootstrap(x: np.ndarray, y: np.ndarray, t: np.ndarray, maxdegree: int, n_bootstraps: int, model, seed: int, test_size=0.2, scale_X=True, scale_t=False, skip_intercept=True, is_scikit=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Method to resample data using the bootstrap algorithm. This method is used to to prepare date to later used in the 
        method bootstrapping() which performs  the actual bootstrapping. 

    Args:
        x ([np.ndarray]): input data in x-direction
        y ([np.ndarray]): input data in y-direction
        t ([np.ndarray]): target data
        maxdegree ([int]): Max degree to create data and fit model on. 
        n_bootstraps ([int]): Number of bootstraps to be performed.
        model ([Callable]): Type of regression model
        seed ([int]): random seed. 
        test_size (float, optional): Size of test. Defaults to 0.2.
        scale_X (bool, optional): scales x. Defaults to True.
        scale_t (bool, optional): scales t. Defaults to False.
        skip_intercept (bool, optional): skips intercept. Defaults to True.
        is_scikit (bool, optional): For use with Lasso and scikit models. Defaults to False.

    Returns:
        Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: MSE_test, MSE_train, bias, variance
    """
    MSE_test = np.zeros(maxdegree)
    MSE_train = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    t_flat = t.ravel().reshape(-1, 1)

    for degree in tqdm(range(1, maxdegree+1), desc=f"Looping through polynomials up to {maxdegree} with {n_bootstraps}: "):
        X = create_X(x, y, n=degree)
        X_train, X_test, t_train, t_test = prepare_data(
            X, t_flat, seed, test_size=test_size, shuffle=True, scale_X=scale_X, scale_t=scale_t, skip_intercept=skip_intercept)
        if not is_scikit:
            t_hat_train, t_hat_test = bootstrapping(
                X_train, t_train, X_test, t_test, n_bootstraps, model)
        else:
            t_hat_train, t_hat_test = bootstrapping_lasso(
                X_train, t_train, X_test, t_test, n_bootstraps, model)

        MSE_test[degree-1] = np.mean([MSE(t_test, t_hat_test[:, strap])
                                     for strap in range(n_bootstraps)])
        MSE_train[degree-1] = np.mean([MSE(t_train, t_hat_train[:, strap])
                                      for strap in range(n_bootstraps)])

        test_arr = [MSE(t_test, t_hat_test[:, strap])
                    for strap in range(n_bootstraps)]

        bias[degree-1] = np.mean(
            (t_test - np.mean(t_hat_test, axis=1, keepdims=True))**2)
        variance[degree-1] = np.mean(np.var(t_hat_test, axis=1, keepdims=True))
    return MSE_test, MSE_train, bias, variance


def bootstrapping(X_train: np.ndarray, t_train: np.ndarray, X_test: np.ndarray, t_test: np.ndarray, n_bootstraps: int, model) -> Tuple[np.ndarray, np.ndarray]:
    """Function to perform the actual bootstrap. Draws n_bootstrap number of resamples. 

    Args:
        X_train (np.ndarray): Input data
        t_train (np.ndarray): train targets
        X_test (np.ndarray): Input data
        t_test (np.ndarray): test targets
        n_bootstraps (int): Number of bootstrap to perform.
        model ([Callable]): Regression model 

    Returns:
        Tuple[np.ndarray,np.ndarray]: Predicted values on train and test.
    """

    t_hat_trains = np.empty((t_train.shape[0], n_bootstraps))
    t_hat_tests = np.empty((t_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        X, t = resample(X_train, t_train)
        t_hat_train = model.fit(
            X, t)
        t_hat_test = model.predict(X_test)
        # Storing predictions
        t_hat_trains[:, i] = t_hat_train.ravel()
        t_hat_tests[:, i] = t_hat_test.ravel()

    return t_hat_trains, t_hat_tests


def bootstrapping_lasso(X_train: np.ndarray, t_train: np.ndarray, X_test: np.ndarray, t_test: np.ndarray, n_bootstraps: int, model) -> Tuple[np.ndarray, np.ndarray]:
    """Function to perform the actual bootstrap, adapted to work with scikit models. Draws n_bootstrap number of resamples. 

    Args:
        X_train (np.ndarray): Input data
        t_train (np.ndarray): train targets
        X_test (np.ndarray): Input data
        t_test (np.ndarray): test targets
        n_bootstraps (int): Number of bootstrap to perform.
        model ([Callable]): Regression model 

    Returns:
        Tuple[np.ndarray,np.ndarray]: Predicted values on train and test.
    """

    t_hat_trains = np.empty((t_train.shape[0], n_bootstraps))
    t_hat_tests = np.empty((t_test.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        X, t = resample(X_train, t_train)
        model.fit(X, t)
        t_hat_train = model.predict(X_train)
        t_hat_test = model.predict(X_test)
        # Storing predictions
        t_hat_trains[:, i] = t_hat_train.ravel()
        t_hat_tests[:, i] = t_hat_test.ravel()
    return t_hat_trains, t_hat_tests


def plot_beta_errors_for_lambdas(summaries_df: pd.DataFrame(), degree: int):
    """Plotting function 

    Args:
        summaries_df (pd.DataFrame): Dataframe containing model summary
        degree ([int]): Model complexity

    Returns:
        plt: Plot. 
    """
    grp_by_coeff_df = summaries_df.groupby(["coeff name"])

    # plt.rcParams["figure.figsize"] = [7.00, 3.50]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i = 0
    for key, item in grp_by_coeff_df:
        df = grp_by_coeff_df.get_group(key)
        # display(df_tmp)
        lambdas = df["lambda"].to_numpy().astype(np.float64)
        beta_values = df["coeff value"].to_numpy().astype(np.float64)
        beta_SE = df["std error"].to_numpy().astype(np.float64)

        # plot beta values
        # plt.plot(lambdas, beta_values, label=f"b{i}")

        plt.plot(lambdas, beta_values, label=fr"$\beta_{i+1}$$\pm SE$")
        # plt.plot(lambdas, beta_values)

        # plot std error
        plt.fill_between(lambdas, beta_values-beta_SE,
                         beta_values+beta_SE, alpha=0.2)

        # 95% CI
        # plt.fill_between(lambdas, CI_lower, CI_upper, alpha = 0.2)
        print("\n\n")
        i += 1

    plt.title(
        f"Plot on Ridge coefficients variation with lambda at degree{degree}")
    plt.xlabel("Lambda values")
    plt.ylabel(r"$\beta_i$ $\pm$ SE")
    plt.xscale("log")
    if degree < 5:
        plt.rcParams["figure.autolayout"] = True
        plt.legend(bbox_to_anchor=(1.05, 1.0))
    # plt.show()
    # plt.tight_layout()
    return fig


def plot_beta_CI_for_lambdas(summaries_df: pd.DataFrame(), degree: int):
    """PLotting function for plotting confidence interval for lambda

     Args:
        summaries_df (pd.DataFrame): Dataframe containing model summary
        degree ([int]): Model complexity

    Returns:
        plt: Plot. 
    """
    grp_by_coeff_df = summaries_df.groupby(["coeff name"])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i = 0
    for key, item in grp_by_coeff_df:
        df = grp_by_coeff_df.get_group(key)
        # display(df_tmp)
        lambdas = df["lambda"].to_numpy().astype(np.float64)
        beta_values = df["coeff value"].to_numpy().astype(np.float64)
        CI_lower = df["CI lower"].to_numpy().astype(np.float64)
        CI_upper = df["CI upper"].to_numpy().astype(np.float64)

        # plot beta values
        # plt.plot(lambdas, beta_values, label=f"b{i}")
        plt.plot(lambdas, beta_values, label=fr"$\beta_{i+1}$ with $CI$")
        # plt.plot(lambdas, beta_values)

        # plot std error
        plt.fill_between(lambdas, CI_lower, CI_upper, alpha=0.2)

        # 95% CI
        # plt.fill_between(lambdas, CI_lower, CI_upper, alpha = 0.2)
        print("\n\n")
        i += 1

    plt.title(
        f"Plot on Ridge coefficients variation with lambda at degree{degree}")
    plt.xlabel("Lambda values")
    plt.ylabel(r"$\beta_i$ with $CI_{95}$")
    plt.xscale("log")
    if degree < 5:
        plt.rcParams["figure.autolayout"] = True
        plt.legend(bbox_to_anchor=(1.05, 1.0))
    # plt.tight_layout()
    # plt.show()
    return fig


def plot_beta_errors(summaary_df: pd.DataFrame(), degree: int, fig=plt.figure()):
    """Plotting function to plot beta errors. 

    Args:
        summaary_df (pd.DataFrame):  Dataframe containing model summary
        degree ([int]): model complexity
        fig ([type], optional): [description]. Defaults to plt.figure().

    Returns:
        plt: Plot
    """

    betas = summaary_df["coeff value"].to_numpy().astype(np.float64)
    SE = summaary_df["std error"].to_numpy().astype(np.float64)

    # Computing x-ticks
    x_ticks_values = np.arange(summaary_df.shape[0])

    ax = plt.axes()
    plt.title(f"Beta error OLS - degree{degree}")
    plt.xlabel(r"$\beta_i$")
    plt.ylabel("Beta values with std error")
    ax.set_xticks(np.arange(summaary_df.shape[0]))
    # ax.set_xticklabels(x_ticks)
    ax.set_xticklabels(rf"$\beta${i+1}" for i in x_ticks_values)
    # plt.gca().margins(x=0)
    # plt.gcf().canvas.draw()
    #tl = plt.gca().get_xticklabels()
    #maxsize = max([t.get_window_extent().width for t in tl])
    # m = 0.2  # inch margin
    #s = maxsize/plt.gcf().dpi*summaary_df.shape[0]+2*m
    #margin = m/plt.gcf().get_size_inches()[0]

    #plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    #plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.errorbar(x_ticks_values, betas, yerr=1.96*SE, fmt='o', ms=4)
    plt.xticks(rotation=90)
    plt.grid()
    # for i, txt in enumerate(x_ticks):
    #    plt.annotate(f"{txt}", (x_ticks_values[i], betas[i]))
    # plt.tight_layout(pad=3.)
    # plt.margins(0.5)
    # plt.show()
    return fig


def cross_val_OLS(k: int, X: np.ndarray, z: np.ndarray, shuffle=False, random_state=None) -> np.ndarray:
    """Function for cross validating on k folds. To be used with OLS. Scales data after split(standarscaler).

    Args:
        k (int): Number of folds
        X (np.ndarray): Design matrix
        z (np.ndarray): target values
        shuffle (boolean): deafault False. 
        random_state(Optional) : Seed value. Defaults to None. 

    Returns:
        np.ndarray: Scores of MSE on all k folds
    """

    model = OLS()

    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    scores_KFold = np.zeros(k)
    z = z.ravel()
    # scores_KFold idx counter
    j = 0
    for train_inds, test_inds in kfold.split(X, z):

        # get all cols and selected train_inds rows/elements:
        xtrain = X[train_inds, :]
        ytrain = z[train_inds]
        # get all cols and selected test_inds rows/elements:
        xtest = X[test_inds, :]
        ytest = z[test_inds]
        # fit a scaler to train_data and transform train and test:
        scaler = StandardScaler()
        scaler.fit(xtrain)
        xtrain_scaled = scaler.transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        xtrain_scaled[:, 0] = 1
        xtest_scaled[:, 0] = 1

        model.fit(xtrain_scaled, ytrain)

        ypred = model.predict(xtest_scaled)
        scores_KFold[j] = MSE(ypred, ytest)
        j += 1

    return scores_KFold


def cross_val(k: int, model: str, X: np.ndarray, z: np.ndarray, degree: int, lmb=None, shuffle=False, random_state=None, scale_t=True) -> np.ndarray:
    """Function for cross validating on k folds. Modified and extended from cross_val_OLS.
        Scales data after split(standarscaler).

    Args:
        k (int): Number of folds
        model (str): Chosen regression model
        X (np.ndarray): Design matrix
        z (np.ndarray): target values
        lmb (Optional): lambda value
        shuffle (boolean): deafault False.
        random_state(Optional) : Seed value. Defaults to None.
        scale_t (boolean) : Wether to scale target values. Default True 


    Returns:
        np.ndarray: Scores of MSE on all k folds
    """

    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    scores_KFold = np.zeros(k)
    # scores_KFold idx counter
    j = 0
    z = z.ravel().reshape(-1, 1)

    for train_inds, test_inds in kfold.split(X, z):

        # get all cols and selected train_inds rows/elements:
        xtrain = X[train_inds, :]
        ztrain = z[train_inds]
        # get all cols and selected test_inds rows/elements:
        xtest = X[test_inds, :]
        ztest = z[test_inds]
        # fit a scaler to train_data and transform train and test:
        data_scaler = StandardScaler()
        data_scaler.fit(xtrain)
        xtrain_scaled = data_scaler.transform(xtrain)
        xtest_scaled = data_scaler.transform(xtest)

        if scale_t == True:
            target_scaler = StandardScaler()
            target_scaler.fit(ztrain)
            ztrain = target_scaler.transform(ztrain)
            ztest = target_scaler.transform(ztest)

        if model == "Ridge":
            model = RidgeRegression(lmb)
            model.fit(xtrain_scaled, ztrain)
        elif model == "Lasso":
            model = lm.Lasso(alpha=lmb, fit_intercept=False,
                             random_state=random_state)
            model.fit(xtrain_scaled, ztrain)
        elif model == "OLS":
            model = OLS(degree=degree)
            model.fit(xtrain_scaled, ztrain)
        else:
            "Provide a valid model as a string(Ridge/Lasso/OLS) "

        zpred = model.predict(xtest_scaled)

        scores_KFold[j] = MSE(zpred, ztest)

        j += 1

    return scores_KFold


def noise_factor(n: int, factor=0.2) -> np.ndarray:
    """Adding noise to whole meshgrid. 

    Args:
        n ([type]): length of meshgrid
        factor (float, optional): The factor of how much noise to be added. Defaults to 0.2.

    Returns:
        np.ndarray: Noise!
    """
    return factor*np.random.randn(n, n)  # Stochastic noise


def MSE_implemented(y_data: np.ndarray, y_model: np.ndarray) -> float:
    """Simple Morten function to compute MSE

    Args:
        y_data ([type]): Truth
        y_model ([type]): predicted values

    Returns:
        [type]: MSE
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def R2_implemented(y_data: np.ndarray, y_model: np.ndarray) -> float:
    """Simple Morten function to compute R2

       Args:
        y_data ([type]): Truth
        y_model ([type]): predicted values

    Returns:
        [type]: R2 score
    """
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


if __name__ == '__main__':
    print("Import this file as a package please!")
