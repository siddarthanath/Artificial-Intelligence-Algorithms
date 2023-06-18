"""
This file creates a Convolutional Neural Network (CNN).
This CNN has been trained and tested on MNIST datasets.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Callable

# Third Party
import torch
import torch.nn as nn

# Private

# -------------------------------------------------------------------------------------------------------------------- #

# Convolutional Neural Network


class CNN(nn.Module):
    """This initialises the CNN."""

    def __init__(self):
        # Allows for multiple inheritance
        super(CNN, self).__init__()
        # Create network
        self.net1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.net2 = nn.Linear(32 * 7 * 7, 10)

    """This applies a forward pass to the input data."""

    def forward(self, x: torch.Tensor):
        # Create network
        out = self.net1(x)
        # Original size: (128, 32, 7, 7) --> New out size: (128, 32*7*7)
        out = out.view(out.size(0), -1)
        # Linear Function (applied to convolution layer output) i.e. pre-activation first hidden layer.
        out = self.net2(out)
        return out

    """This fits the CNN to the training set and evaluates the FFNN on the validation set."""

    def fit(
        self,
        num_epochs: int,
        patience: int,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        loss_fn: Callable,
        opt: Callable,
    ):
        # Train model
        best_val_loss = float("inf")
        epochs_no_improve = 0
        # Iterate through number of epochs
        for epoch in range(num_epochs):
            # Iterate through entire collection (per batch)
            for images, labels in train_loader:
                # Clear gradients with respect to parameters
                opt.zero_grad()
                # Forward pass
                outputs = self(images)
                # Calculate loss
                loss = loss_fn(outputs, labels)
                # Backward pass
                loss.backward()
                # Parameter update
                opt.step()
            # Evaluate model (NOTE: In the evaluation phase, the model parameters do not need updating!)
            with torch.no_grad():
                # Initialise validation loss
                val_loss = 0
                # Initialise secondary metric
                correct = 0
                # Iterate through entire collection (per batch)
                for images, labels in val_loader:
                    # Forward pass
                    outputs = self(images)
                    # Obtain predictons
                    _, preds = torch.max(outputs.data, dim=1)
                    # Total correct predictions
                    correct += (preds == labels).sum()
                    # Calculate loss
                    loss = loss_fn(outputs, labels)
                    # Accumulate loss
                    val_loss += loss.item()
                # Record secondary metric
                val_acc_score = 100 * correct / len(val_loader.dataset)
                # Print loss and accuracy
                print(
                    f"Epoch: {epoch} | Validation loss: {val_loss} | Validation accuracy: {val_acc_score}"
                )
                # If the validation loss is at a new minimum, save the model
                if val_loss < best_val_loss:
                    torch.save(self.state_dict(), "best_model.pth")
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                # If the validation loss is not improving for certain patience, stop training!
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        break

    """This predicts on the test dataset using the optimal CNN obtained from fit."""

    def predict(self, test_loader: torch.utils.data.dataloader.DataLoader):
        # Load saved model (NOTE: Due to the fit, we can assume that a model is always saved i.e. has improved through training.)
        try:
            self.load_state_dict(torch.load("best_model.pth"))
        except FileNotFoundError as e:
            print(f"Error: {e} - no model has been saved!")
        # Store predictions for each batch
        all_preds = torch.Tensor()
        for images, _ in test_loader:
            # Forward pass
            outputs = self(images)
            # Obtain predictons
            _, preds = torch.max(outputs.data, dim=1)
            # Store predictions
            all_preds = torch.cat((all_preds, preds), 0)
        # Return predictions
        return all_preds
