import time
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple


# Setting global variables
INPUT_DATA = "../data/input_data/"  # Path for input data
REPORT_DATA = "../data/report_data/"  # Path for data ment for the report
REPORT_FIGURES = "../figures/"  # Path for figures ment for the report
EX_A = "EX_A_"; EX_B = "EX_B_"; EX_C = "EX_C_"; EX_D = "EX_D_"; EX_E = "EX_E_"; EX_F = "EX_F_"

# Common methods

def learning_rate_upper_limit(X_train):
    XT_X = X_train.T @ X_train
    H = (2./X_train.shape[0]) * XT_X # The Hessian is the second derivate
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
    for y in range(0,img.shape[0], ySteps):
        for x in range(0,img.shape[1], xSteps):
            y_from = y; 
            y_to = y+ySteps; 
            x_from = x; 
            x_to = x+xSteps; 
            img_patch = img[y_from:y_to, x_from:x_to]        
            patches.append(img_patch)

    return patches

def patches_to_img(patches, ySteps, xSteps, nYpatches, nXpatches, plotImage=False):
    img = np.zeros((ySteps*nYpatches, xSteps*nXpatches))
    i = 0
    for y in range(0,img.shape[0], ySteps):
        for x in range(0,img.shape[1], xSteps):
            y_from = y; 
            y_to = y+ySteps; 
            x_from = x; 
            x_to = x+xSteps; 
            img[y_from:y_to, x_from:x_to] = patches[i]         
            i += 1
    
    if plotImage:
        plt.imshow(img, cmap='gray')
        plt.title("Reconstructed img")
        plt.show()
    return img

def plotTerrainPatches(patches, nYpatches, nXpatches, plotTitle="Terrain patches"):
    # Plotting terrain patches)
    fig, ax = plt.subplots(nYpatches, nXpatches,figsize=(4,10))
    i=0
    for y in range(nYpatches):
        for x in range(nXpatches):
            ax[y,x].title.set_text(f"Patch{i}")
            ax[y,x].set_xlabel("X"); ax[y,x].set_ylabel("Y")
            ax[y,x].imshow(patches[i], cmap='gray')
            i+=1
    
    fig.suptitle(f"{plotTitle}") # or plt.suptitle('Main title')
    plt.tight_layout()
    
    return fig
    plt.show()

def createTerrainData(terrain, includeMeshgrid=True):
    z = np.array(terrain) 
    x = np.arange(0, z.shape[1])
    y = np.arange(0, z.shape[0])
    if includeMeshgrid:
        x, y = np.meshgrid(x,y)
    return x,y,z


if __name__ == '__main__':
    print("Import this file as a package please!")
