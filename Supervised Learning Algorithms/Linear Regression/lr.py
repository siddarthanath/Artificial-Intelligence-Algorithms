"""
This file runs Linear Regression.
Note: The boston-filtered csv file and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Dict, Union

# Third Party
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

# Private


# -------------------------------------------------------------------------------------------------------------------- #
class LinearRegression:
    def __init__(self, w: np.array, shuffle: bool, train_ratio: float = 0.66):
        """
        Initialise Linear Regression.
        Note that Linear Regression does not have any hyperparameters, unless if trying to calculate the parameters via
        gradient descent (where then we could use cross validation to find optimal learning rate).
        Args:
            w:
                Weights of Linear Regression model.
            shuffle:
                Whether to shuffle the dataset or not.
            train_ratio:
                The fraction of train data to have.
        Returns:
            N/A
        """
        self.w = None
        self.shuffle = shuffle
        self.train_ratio = train_ratio

    def fit(self, X_train: np.array, y_train: np.array, library: bool):
        """
        Fit Linear Regression model (typically on train data).
        Args:
            X_train:
                The feature matrix.
            y_train:
                The target values.
            library:
                Whether to use inbuilt-library for more precise calculation
        Returns:
            An array of containing the weights.
        """
        # Return weights
        if library is True:
            self.w = np.linalg.lstsq(X_train, y_train, rcond=None)[0].reshape(-1, 1)
        else:
            self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    def predict(self, X_test: np.array) -> np.array:
        """
        Return predictions (typically on test data).
        Args:
            w:
                An array containing weights of fitted Linear Regression model.
            X_test:
                The test data to calculate predictions on.
        Returns:
            An array of predictions.
        """
        self.check_is_fitted()
        return X_test @ self.w

    @staticmethod
    def scores(
            y_values: np.array, y_values_pred: np.array
    ) -> Dict[str, Union[float, int]]:
        """
        Calculate generalisation scores.
        Args:
            y_values:
                The discrete labels or continuous quantity (of the test data usually).
            y_values_pred:
                The predicted discrete labels or continuous quantity (of the test data usually).
        Returns:
            A dictionary of float values.
        """
        return {
            "MSE": sm.mean_squared_error(y_true=y_values, y_pred=y_values_pred),
            "MAE": sm.mean_absolute_error(y_true=y_values, y_pred=y_values_pred),
        }

    def plot_decision_boundary(self, X: np.array, y_true: np.array, y_pred: np.array):
        """
        Plot decision boundary of Linear Regression.
        Args:
            X:
                The feature matrix.
            y_true:
                The target values.
            y_pred:
                The predicted target values.
        Returns:
            A matplotlib image.
        """
        # Check that dataset has only two features (otherwise visualisation is not possible)
        if X.shape[1] != 2:
            raise ValueError(
                "It is only possible to visualise a dataset that has only 2 features."
            )
        # If bias has been added, only visualise second column, else visualise both columns
        if np.allclose(X[:, 0], np.ones(X.shape[0])):
            # Plot image
            plt.figure()
            # True datapoints
            plt.scatter(X[:, 1], y_true, label= "True Values")
            # Predicted datapoints
            plt.plot(X[:, 1], y_pred, "--", label= "Predicted Values")
            plt.title(f"Linear Regression Model")
            plt.legend()
            plt.show()
        else:
            # Plot image
            plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            ax.grid()
            # True datapoints
            ax.scatter(X[:, 0], X[:, 1], y_true)
            ax.set_title("Linear Regression Model")
            # Set axes label
            ax.set_xlabel("Attribute 1", labelpad=20)
            ax.set_ylabel("Attribute 2", labelpad=20)
            ax.set_zlabel("Output", labelpad=20)
            # Predicted datapoints
            ax.plot3D(X[:, 0], X[:, 1], y_pred)
            plt.show()

    def check_is_fitted(self):
        """
        Check if Linear Regression model is fitted before prediction occurs.
        Args:
            N/A
        Returns:
            ValueError (if model has not been fitted).
        """
        if self.w is None:
            raise ValueError("Linear Regression model has not been fitted yet!")

    # def lr_all_attr(self, X: np.array, num_iter: int, partition: bool, cross_val: bool):
    #     """
    #     Run Linear Regression model (on potentially different shuffled parts of the dataset).
    #     Args:
    #         X:
    #             Input data (including features and target).
    #         num_iter:
    #             Number of times to run linear regression model on a different set of data points.
    #         cross_val"
    #             Whether to perform cross validation on that particular set of data and return scores.
    #     Returns:
    #         A dataframe of generalisation scores.
    #     """
    #     # Store errors
    #     full_train_error = []
    #     full_test_error = []
    #     # Find average MSE of train and test error
    #     for _ in range(num_iter):
    #         # Obtain train and test data
    #         X_train, y_train, X_test, y_test = self.train_test_split(X=X, partition=partition)
    #         # Execute cross validation
    #         if cross_val is True:
    #
    #         # Perform linear regression
    #         w_full = ls_lr(X=X_train_full, y=y_train_full)[0].reshape(-1, 1)
    #         # Calculate MSE
    #         full_train_error.append(mse(y_train_full, X_train_full @ w_full))
    #         full_test_error.append(mse(y_test_full, X_test_full @ w_full))
    #     full_train_error_avg = np.mean(full_train_error, axis=0)
    #     full_test_error_avg = np.mean(full_test_error, axis=0)
    #     full_train_std_avg = np.std(full_train_error, axis=0)
    #     full_test_std_avg = np.std(full_test_error, axis=0)
    #     return full_train_error_avg, full_test_error_avg, full_train_std_avg, full_test_std_avg

    def train_test_split(self, X: np.array, partition: bool):
        """
        Creates train and test dataset.
        Args:
            X:
                Input data (including features and target).
            partition:
                Whether to split the dataset into feature matrix and target or not (for train and test).
        Returns:
            train, test:
                Train and test set (entire set or split into feature matrix and target).
        """
        # Store output data
        if self.shuffle:
            np.random.shuffle(X)
        # Create train and test data
        train = X[: int(self.train_ratio * len(X)), :]
        test = X[len(train) :, :]
        # Create X and y train and test data
        if partition is True:
            X_train = train[:, :-1]
            y_train = train[:, -1].reshape(-1, 1)
            X_test = test[:, :-1]
            y_test = test[:, -1].reshape(-1, 1)
            return X_train, y_train, X_test, y_test
        else:
            return train, test