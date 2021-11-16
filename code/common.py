import time
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple
import pickle
import pandas as pd


# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/"  # Path for data ment for the report
REPORT_FIGURES = "../figures/"  # Path for figures ment for the report
EX_A = "EX_A_"
EX_B = "EX_B_"
EX_C = "EX_C_"
EX_D = "EX_D_"
EX_E = "EX_E_"
EX_F = "EX_F_"

# Common methods


def learning_rate_upper_limit(X_train):
    XT_X = X_train.T @ X_train
    H = (2./X_train.shape[0]) * XT_X  # The Hessian is the second derivate
    # Picking the largest eigenvalue of the Hessian matrix to use as a guide for determain upper limit for learning rate
    lr_upper_limit = 2./np.max(np.linalg.eig(H)[0])
    print(f"Upper limit learing rate: {lr_upper_limit}")
    return lr_upper_limit


# Methods below are reused from from project1
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


def create_img_patches(img, ySteps, xSteps):
    patches = []
    for y in range(0, img.shape[0], ySteps):
        for x in range(0, img.shape[1], xSteps):
            y_from = y
            y_to = y+ySteps
            x_from = x
            x_to = x+xSteps
            img_patch = img[y_from:y_to, x_from:x_to]
            patches.append(img_patch)

    return patches


def patches_to_img(patches, ySteps, xSteps, nYpatches, nXpatches, plotImage=False):
    img = np.zeros((ySteps*nYpatches, xSteps*nXpatches))
    i = 0
    for y in range(0, img.shape[0], ySteps):
        for x in range(0, img.shape[1], xSteps):
            y_from = y
            y_to = y+ySteps
            x_from = x
            x_to = x+xSteps
            img[y_from:y_to, x_from:x_to] = patches[i]
            i += 1

    if plotImage:
        plt.imshow(img, cmap='gray')
        plt.title("Reconstructed img")
        plt.show()
    return img


def plotTerrainPatches(patches, nYpatches, nXpatches, plotTitle="Terrain patches"):
    # Plotting terrain patches)
    fig, ax = plt.subplots(nYpatches, nXpatches, figsize=(4, 10))
    i = 0
    for y in range(nYpatches):
        for x in range(nXpatches):
            ax[y, x].title.set_text(f"Patch{i}")
            ax[y, x].set_xlabel("X")
            ax[y, x].set_ylabel("Y")
            ax[y, x].imshow(patches[i], cmap='gray')
            i += 1

    fig.suptitle(f"{plotTitle}")  # or plt.suptitle('Main title')
    plt.tight_layout()

    return fig
    plt.show()


def createTerrainData(terrain, includeMeshgrid=True):
    z = np.array(terrain)
    x = np.arange(0, z.shape[1])
    y = np.arange(0, z.shape[0])
    if includeMeshgrid:
        x, y = np.meshgrid(x, y)
    return x, y, z


def save_model(model, path_filename):
    "saving the medel as .pkl filetype"
    with open(path_filename, 'wb') as outp:  # Overwrites existing .pkl file.
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def load_model(path_filename):
    "Loading a .pkl filetype"
    with open(path_filename, 'rb') as inp:
        model = pickle.load(inp)
    return model


# OLS from project1
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


if __name__ == '__main__':
    print("Import this file as a package please!")
