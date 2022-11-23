"""
This file runs K-Nearest-Neighbours.
Note: The diabetes csv file and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Union, Dict
from collections import Counter

# Third Party
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

# Private


# -------------------------------------------------------------------------------------------------------------------- #
class KNN:
    def __init__(self, X: np.array, y: np.array, k: int, ml_task: str):
        """
        Initialise KNN classification or regression model.
        Note, for classification, that this assumes the labels are in sequential order.
        Args:
            X:
                Typically the train data.
                Note that KNN does not have parameters and so there is no concept of "fitting".
            y:
                The labels of the train data.
            k:
                The number of neighbours to consider.
            ml_task:
                Either classification or regression.
        Returns:
            N/A
        """
        self.X = X
        self.y = y
        self.k = k
        self.ml_task = (
            ml_task if ml_task == "classification" or ml_task == "regression" else None
        )

    def predict(self, X_other: np.array) -> np.array:
        """
        Predict labels or values for both classification and regression.
        Note that we use KDTree for fast neighbour implementation - one can also do this from scratch.
        Args:
            X_other:
                Typically the test data to find distances against each train data.
        Returns:
            An array of label predictions.
        """
        # Store predictions
        y_pred = np.zeros(X_other.shape[0])
        # Calculate closest neighbour indices
        tree = KDTree(self.X)
        # Obtain indices of closest k neighbours
        indices = tree.query(X_other, k=self.k, return_distance=False)
        # Obtain labels of k nearest neighbours
        for i in range(len(indices)):
            nn_values = self.y[indices[i]]
            # Assign target label or average value
            if self.ml_task == "classification":
                # Count occurrences of each label
                label_counts = Counter(nn_values)
                # Find maximum number of label occurrences
                max_counts = max(label_counts.values())
                pred_labels = [k for k, v in label_counts.items() if v == max_counts]
                # If one label has most occurrences, pick this as prediction, else randomly pick between highest
                if len(pred_labels) == 1:
                    y_pred[i] = pred_labels[0]
                else:
                    y_pred[i] = np.random.choice(pred_labels)
            elif self.ml_task == "regression":
                y_pred[i] = np.mean(nn_values)
            else:
                raise ValueError("Correct ml_task has not been provided.")
        return y_pred

    def scores(
        self, y_values: np.array, y_values_pred: np.array
    ) -> Union[float, Dict[str, Union[float, int]]]:
        """
        Calculate generalisation scores.
        Args:
            y_values:
                The discrete labels or continuous quantity (of the test data usually).
            y_values_pred:
                The predicted discrete labels or continuous quantity (of the test data usually).
        Returns:
            A float value.
        """
        if self.ml_task == "classification":
            f_1_score = sm.f1_score(y_true=y_values, y_pred=y_values_pred)
            acc_score = sm.accuracy_score(y_true=y_values, y_pred=y_values_pred)
            precision_score = sm.precision_score(y_true=y_values, y_pred=y_values_pred)
            recall_score = sm.recall_score(y_true=y_values, y_pred=y_values_pred)
            error_score = 1 - acc_score
            return {
                "F1": f_1_score,
                "Accuracy": acc_score,
                "Error": error_score,
                "Precision": precision_score,
                "Recall": recall_score,
            }
        elif self.ml_task == "regression":
            return {
                "MSE": sm.mean_squared_error(y_true=y_values, y_pred=y_values_pred),
                "MAE": sm.mean_absolute_error(y_true=y_values, y_pred=y_values_pred),
            }
        else:
            raise ValueError("Correct ml_task has not been provided.")

    def plot_decision_regions(self):
        """
        Plot decision boundary of KNN.
        Args:
            N/A
        Returns:
            A matplotlib image.
        """
        # Check that dataset has only two features (otherwise visualisation is not possible)
        if self.X.shape[1] != 2 or self.ml_task != "classification":
            raise ValueError(
                "It is only possible to visualise a dataset that has only 2 features."
            )
        # Step size in the mesh
        h = 0.02
        # Create color maps
        cmap_light = mc.ListedColormap(["blue", "red"])
        cmap_bold = mc.ListedColormap(["white", "black"])
        # Plot the decision boundary
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(X_other=np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        # Plot also the training points
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap_bold)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"2-Class classification (k = {self.k})")
        plt.show()


# Create main function to run via terminal
def main():
    # Obtain csv files
    main_df = pd.read_csv("diabetes.csv")
    # Ask for number of neighbours
    print("Enter number of neighbours required for this dataset:")
    num_neighbours = int(input())
    # Ask what machine learning task
    print("Enter machine learning task for this dataset:")
    ml_task = str(input()).lower()
    # Ask if user wants to shuffle the dataset
    print("Enter True or False (if you want to shuffle the dataset):")
    bool_shuffle = eval(input())
    # Put data into feature matrix and target
    X, y = main_df.iloc[:, :-1], main_df.iloc[:, -1].to_numpy()
    # Scale feature matrix to prevent distance problems
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0))
    # Change df to array
    X = X.to_numpy()
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=bool_shuffle
    )
    # Create KNN model
    knn_model = KNN(X=X_train, y=y_train, k=num_neighbours, ml_task=ml_task)
    # Predict neighbours for test set
    y_values_pred = knn_model.predict(X_other=X_test)
    # Calculate scores
    main_df_scores = knn_model.scores(y_values=y_test, y_values_pred=y_values_pred)
    # Print score
    for count, key, value in enumerate(main_df_scores.items()):
        print(f"{count+1}) The {key} score for this dataset is: {value}. \n")


# Execute code via terminal
if __name__ == "__main__":
    main()
