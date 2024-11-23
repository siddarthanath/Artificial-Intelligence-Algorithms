"""
This file runs Linear Regression.
Note: The data file and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import List, Optional

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

# Private
from base import SLBase


# -------------------------------------------------------------------------------------------------------------------- #

class LinearRegression(SLBase):

    def __init__(self, column_names: Optional[List[str]]):
        self.beta = None
        self.beta_history = []
        self.loss_history = []
        self.features = column_names

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X[:, 0] == np.ones(len(X))
        self.beta = np.linalg.solve((X.T @ X), X.T @ y)

    def train(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.001, epsilon: float = 0.001,
              batch_size: int = 32, num_epochs: int = 10):
        assert np.array_equal(X[:, 0], np.ones(len(X)))
        assert len(X) == len(y)
        assert len(X) >= 32
        # Initialise gradient (NOTE: Only for assignment purposes)
        grad = np.inf
        # Obtain indices to slice the data
        indices = np.arange(0, len(X))
        # Initialise weights
        self.beta = np.zeros(X.shape[1])
        # Store gradient
        self.beta_history.append(self.beta.copy())
        # Obtain batch
        for _ in range(num_epochs):
            # Calculate train loss
            train_loss = 0
            # Stochastic
            np.random.shuffle(indices)
            batch_generator = self._get_batches(X, y, indices, batch_size)
            # Batch Gradient Descent
            for X_batch, y_batch in batch_generator:
                # Calculate predictions
                y_batch_pred = self.predict(X_batch)
                # Calculate training loss
                train_loss += self.scores(y_batch, y_batch_pred)
                # Calculate gradient
                grad = 2 / len(X_batch) * X_batch.T @ (y_batch_pred - y_batch)
                # Update parameters
                self.beta -= alpha * grad
            train_loss /= int(np.ceil(len(X) / batch_size))
            print(f"Epoch {_ + 1}: Train Loss = {train_loss}")
            # Store gradient
            self.beta_history.append(self.beta)
            # Store train loss
            self.loss_history.append(train_loss)
            # Check for early convergence
            if np.linalg.norm(grad) <= epsilon:
                break

    @staticmethod
    def _get_batches(X: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int = 32):
        # Loop over batches
        for j in range(0, len(X), batch_size):
            # Obtain batch
            batch_indices = indices[j:j + batch_size]
            yield X[batch_indices], y[batch_indices]

    def predict(self, X: np.ndarray):
        assert np.array_equal(X[:, 0], np.ones(len(X)))
        return X @ self.beta

    @staticmethod
    def scores(y_train: np.ndarray, y_test: np.ndarray):
        assert len(y_train) == len(y_test)
        return 1 / len(y_train) * np.sum((y_train - y_test) ** 2)

    def plot_loss(self):
        if self.loss_history is not None:
            plt.plot(np.arange(1, len(self.loss_history) + 1), self.loss_history, '-o', label='Training Loss')
            plt.ylabel(f'Error')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()
        else:
            raise ValueError("No loss history exists as algorithm has not been trained!")

    def _animate_regression(self, X, y):
        assert np.array_equal(X[:, 0], np.ones(len(X)))
        if self.beta_history is not None:
            fig, ax = plt.subplots()
            plt.scatter(X[:, 1], y, color='blue', alpha=0.5, label='Data points')

            def update(frame):
                plt.cla()  # Clear the current axis
                plt.scatter(X[:, 1], y, color='blue', alpha=0.5, label='Data points')
                # Get x values for plotting
                x_plot = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
                # Stack ones with x_plot
                X_plot = np.column_stack([np.ones(100), x_plot])
                # Calculate corresponding y values using current beta
                y_plot = X_plot @ self.beta_history[frame]
                plt.plot(x_plot, y_plot, 'r-', label=f'Regression line (epoch {frame})')
                plt.xlabel('X')
                plt.ylabel('y')
                plt.title('Linear Regression Animation')
                plt.legend()

            anim = animation.FuncAnimation(fig, update,
                                           frames=len(self.beta_history),
                                           interval=200, repeat=False)
            # Save animation
            anim.save('regression.gif', writer='pillow')
            plt.show()
            return anim
        else:
            raise ValueError("No beta history as algorithm has not been trained!")


def main():
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    # Create X (between -3 and 3)
    X_raw = np.random.uniform(-3, 3, n_samples)
    # Create y with multiple non-linear components and clusters
    y = (2 * X_raw + 1 +
         5 * np.sin(2 * X_raw) +
         0.5 * X_raw ** 2 +
         np.random.normal(0, 3, n_samples))
    # Add some outliers
    outlier_idx = np.random.choice(n_samples, size=10)
    y[outlier_idx] += np.random.normal(10, 5, 10)
    # Standardize X
    X_scaled = (X_raw - np.mean(X_raw)) / np.std(X_raw)
    # Create final X with ones column
    X = np.column_stack([np.ones(n_samples), X_scaled])
    # Create and train model
    model = LinearRegression(column_names=None)
    model.train(X, y,
                alpha=0.01,
                epsilon=0.001,
                batch_size=32,
                num_epochs=50)
    # Plot training loss
    model.plot_loss()
    # Create animation (assuming single feature for visualization)
    if X_scaled.ndim == 1:
        model._animate_regression(X, y)
    else:
        print("Animation only works for single feature datasets")


if __name__ == "__main__":
    main()
