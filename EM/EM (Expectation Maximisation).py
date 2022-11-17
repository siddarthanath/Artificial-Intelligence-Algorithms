"""
This file runs Q2.
Note: The binary digits text file and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library
from typing import Tuple
# 3rd Party
import numpy as np
from scipy.special import beta, betaln


# Private
# -------------------------------------------------------------------------------------------------------------------- #


def model_1(X: np.array, pixel_value: float) -> Tuple[float, float]:
    """
    Calculate posterior probability of model 1.
    Args:
        X:
            An array containing image data.
        pixel_value:
            The value of the pixel parameter used to generate the pixels in each of the images.
    Returns:
        Tuple containing posterior probability (with and without log).
    """
    # Assertions
    assert isinstance(pixel_value, float)
    # Obtain size of array
    N, D = X.shape
    # Return probability and log probability
    return (1 / 2) ** (D * N), -np.log(2) * D * N


def model_2(X: np.array) -> Tuple[float, float]:
    """
    Calculate posterior probability of model 2.
    Args:
        X:
            An array containing image data.
    Returns:
        Tuple containing posterior probability (with and without log).
    """
    # Obtain size of array
    N, D = X.shape
    # Calculate W
    W = np.sum(X)
    # Return log probability
    return beta(W + 1, D * N - W + 1), betaln(W + 1, D * N - W + 1)


def model_3(X: np.array) -> Tuple[float, float]:
    """
    Calculate posterior probability of model 3.
    Args:
        X:
            An array containing image data.
    Returns:
        Tuple containing posterior probability (with and without log).
    """
    # Obtain size of array
    N, D = X.shape
    # Calculate W_D
    W_D = np.sum(X, axis=0)
    # Return log probability
    return np.prod(beta(W_D + 1, N - W_D + 1)), np.sum(betaln(W_D + 1, N - W_D + 1))


def normalised_log_proba(X: np.array):
    """
    Given an array of probabilities, calculate the normalised log probability.
    Args:
        X:
            An array of probabilities.
    Returns:
        An array of normalised log probabilities.
    """
    c = X.max()
    new_val = c + np.log(np.sum(np.exp(X - c)))
    return np.exp(X - new_val)


# Create main function to run via terminal
def main():
    # Obtain file name
    data = "binarydigits.txt"
    # Load data
    X = np.loadtxt(data)
    # Calculate unnormalised log probabilities
    _, post_log_1 = model_1(X=X, pixel_value=0.5)
    _, post_log_2 = model_2(X=X)
    _, post_log_3 = model_3(X=X)
    # Calculate normalised log probabilities
    unnormalised_log_probas = np.array([post_log_1, post_log_2, post_log_3])
    norm_log_probas = normalised_log_proba(unnormalised_log_probas)
    # Print values for Model 1
    print(f"For Model 1: Normalised Posterior probability = {norm_log_probas[0]} | Log Posterior Probability = "
          f"{post_log_1}")
    # Print values for Model 2
    print(f"For Model 2: Normalised Posterior probability = {norm_log_probas[1]} | Log Posterior Probability = "
          f"{post_log_2}")
    # Print values for Model 1
    print(f"For Model 3: Normalised Posterior probability = {norm_log_probas[2]} | Log Posterior Probability = "
          f"{post_log_3}")


# Execute code via terminal
if __name__ == "__main__":
    main()
