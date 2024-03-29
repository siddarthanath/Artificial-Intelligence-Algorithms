"""
This file runs Discriminant Analysis.
Note: The data file and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Union, Dict

# Third Party
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as sm

# Private


# -------------------------------------------------------------------------------------------------------------------- #
# Third Party
class GDA:
    def __init__(self, pi_weights, mu_weights, cov_weights, optim_type):
        """
        Initialise DA classification model. This model specifically uses a Gaussian likelihood.
        """
        self.pi = pi_weights
        self.mu = mu_weights
        self.cov = cov_weights
        self.optim_type = optim_type

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """This function returns a fitted DA model.

        Args:
            X (pd.DataFrame): The train feature matrix.
            y (pd.Series): The target.
            optim_type (str): Whether to automate or use gradient descent to solve.

        Raises:
            NotImplementedError: ...
            ValueError: If optimisation type is invalid.
        """
        # Calculate number of unique classes
        num_classes = np.unique(y)
        # Choose optimisation process
        if self.optim_type == "auto":
            # Prior probability on which Gaussian corresponds to which class
            self.pi = {class_label: sum(y==class_label)/len(y) for class_label in num_classes}
            # Likelihood class mean
            self.mu = {class_label: 1/sum(y==class_label) * X[y==class_label].sum(axis=0) for class_label in num_classes}
            # Likelihood class covariance
            self.cov = {class_label: 1/sum(y==class_label) * (X[y==class_label] - self.mu[class_label]).T @ (X[y==class_label] - self.mu[class_label]) for class_label in num_classes}
        elif self.optim_type == "em":
            raise NotImplementedError
        else:
            raise ValueError("Optimisation algorithm type not recognisable.")


    def predict(self, X_test: np.array) -> np.array:
        """Returns prediction from the test set.

        Args:
            X_test (np.array): The test feature matrix.

        Returns:
            np.array: Predicted class values.
        """
        # Return optimal parameters
        if all([self.pi, self.mu, self.cov]):
            # Calcualate posterior
            posterior_x = pd.DataFrame([self.posterior(x=x) for x in X_test])
            # Calculate maximum posterior probability
            return posterior_x.idxmax(axis=1)
        else:
            return TypeError("The model parameters have not been fitted.")
        
    def scores(
        self, y_true: np.array, y_pred: np.array
    ) -> Union[float, Dict[str, Union[float, int]]]:
        """Calculates classification metrics.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predict labels.

        Returns:
            Union[float, Dict[str, Union[float, int]]]: The classification metrics
        """
        if self.ml_task == "classification":
            f_1_score = sm.f1_score(y_true=y_true, y_pred=y_pred)
            acc_score = sm.accuracy_score(y_true=y_true, y_pred=y_pred)
            precision_score = sm.precision_score(y_true=y_true, y_pred=y_pred)
            recall_score = sm.recall_score(y_true=y_true, y_pred=y_pred)
            error_score = 1 - acc_score
            return {
                "F1": f_1_score,
                "Accuracy": acc_score,
                "Error": error_score,
                "Precision": precision_score,
                "Recall": recall_score,
            }
        
    def posterior(self, x: pd.Series) -> Dict[str, float]:
        """Calculates the posterior function, for particular datapoint, over all classes.

        Args:
            x (pd.Series): A data point.

        Returns:
            Dict[str, float]: The key as class label and value as posterior probability.
        """
        # Calculate multivariate normal for each class
        class_posterior = {}
        n = len(self.mu)
        for class_label in self.mu.keys():
            sigma_det = np.linalg.det(self.cov[class_label])
            sigma_inv = np.linalg.inv(self.cov[class_label])
            normalization_factor = (2 * np.pi)**n * sigma_det
            x_mu = np.matrix(x - self.mu[class_label])
            exponent = np.exp(-0.5 * x_mu * sigma_inv * x_mu.T)
            multivariate_normal = (1.0 / np.sqrt(normalization_factor)) * exponent
            class_posterior[class_label] = self.pi * multivariate_normal
        class_posterior[class_label] = {class_label: posterior/np.sum(class_posterior[class_label] for class_label in class_posterior.keys()) for class_label, posterior in class_posterior.items()}
        return class_posterior
    
    def log_likelihood(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculates the log likelihood function, over all datapoints, over all classes.

        Args:
            X (pd.DataFrame): _description_
            y (pd.Series): _description_

        Returns:
            float: _description_
        """
        # Store number of classes
        num_classes = np.unique(y)
        # Accumulate log likelihood
        ll_total = 0
        # Loop over each data point
        for feature, target in zip(X, y):
            first_term = np.log(self.pi_weights[target])
            second_term = np.log(self.multivariate_normal(n=num_classes, class_label=target, mu=self.mu, cov=self.cov, x=feature))
            ll_total += first_term + second_term
        return ll_total
    
    @staticmethod
    def multivariate_normal(n: int, class_label: int, mu, cov, x):
        sigma_det = np.linalg.det(cov[class_label])
        sigma_inv = np.linalg.inv(cov[class_label])
        normalization_factor = (2 * np.pi)**n * sigma_det
        x_mu = np.matrix(x - mu[class_label])
        exponent = np.exp(-0.5 * x_mu * sigma_inv * x_mu.T)
        return (1.0 / np.sqrt(normalization_factor)) * exponent
    
    def plot_ll(values):
        plt.plot(np.arange(1, len(values)+1, 1), values, "-o")
        plt.title("Cumulative log-likelihood of GDA")
        plt.xlabel("Iteration")
        plt.ylabel("Log-likelihood value")
    
    def plot_points(x):
        c = "bgr"
        m = "xos"
        for i, point in enumerate(x):
            N = point.shape[0]
            nplot = min(N, 30)
            plt.plot(point[:nplot, 0], point[:nplot, 1], c[i] + m[i])


    def plot_contours(x, y, u, sigma):
        nclasses = len(u)
        c = "bgr"
        m = "xos"
        for i in range(nclasses):
            xx, yy = np.meshgrid(x, y)
            xy = np.c_[xx.ravel(), yy.ravel()]
            sigma_inv = np.linalg.inv(sigma)
            z = np.dot((xy - u), sigma_inv)
            z = np.sum(z * (xy - u), axis=1)
            z = np.exp(-0.5 * z)
            z = z / (2 * np.pi * np.linalg.det(sigma) ** 0.5)
            prob = z.reshape(xx.shape)
            cs = plt.contour(xx, yy, prob, colors=c[i])

# Create main function to run via terminal
def main():
    # Ask for covariance
    print("Enter covariance for each class:")
    cov = list((input()))
    # Ask for covariance
    print("Enter mean for each class:")
    mu = list((input()))
    # Ask for covariance
    print("Enter pi for each class:")
    pi = np.array((input()))
    # Create pseudo data from user chosen parameters
    ngrid = 200
    n_samples = 100
    x = [] 
    labels = [] 
    nclasses = len(mu)
    for i in range(nclasses):
        x.append(np.random.multivariate_normal(mu[i], cov[i], n_samples))
        labels.append([i] * n_samples)
    # Create 2D visual plot of data
    points = np.vstack(x)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_range = np.linspace(x_min - 1, x_max + 1, ngrid)
    y_range = np.linspace(y_min - 1, y_max + 1, ngrid)
    xx, yy = np.meshgrid(x_range, y_range)
    X = np.vstack(x)
    y = np.hstack(labels)
    # Create GDA model (via MLE)
    model = GDA(pi_weights=None, mu_weights=None, cov_weights=None, optim_type="auto")
    model.fit(X=X, y=y)
    # Plot results
    plt.figure()
    model.plot_points(x)
    plt.axis("square")
    plt.tight_layout()
    plt.title("2D Visual Plot of Multivariate Normal Distribution")
    plt.show()
    plt.figure()
    model.plot_points(x)
    model.plot_contours(x, y, model.mu, model.cov)
    plt.axis("square")
    plt.tight_layout()
    plt.title("Gaussian Cluster Analysis")
    plt.show()


# Execute code via terminal
if __name__ == "__main__":
    main()
