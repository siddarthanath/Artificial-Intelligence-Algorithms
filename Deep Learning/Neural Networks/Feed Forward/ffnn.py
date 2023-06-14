"""
This file creates a Feed=-Forward Neural Network.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Callable

# Third Party
import torch
import torch.nn as nn

# Private

# -------------------------------------------------------------------------------------------------------------------- #

# Feed-Forward Neural Network


class FFNN(nn.Module):
    """This initialises the FFNN."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # Allows for multiple inheritance
        super(FFNN, self).__init__()
        # Create network (NOTE: This is currently a 1 Layer FFNN - we can add more Linear() layers to create a deeper network.)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    """This applies a forward pass to the input data."""

    def forward(self, x: torch.Tensor):
        # Create network
        out = self.net(x)
        return out

    """This fits the FFNN to the training set and evaluates the FFNN on the validation set."""

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
                outputs = self(images.view(images.shape[0], -1))
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
                    outputs = self(images.view(images.shape[0], -1))
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

    """This predicts on the test dataset using the optimal FFNN obtained from fit."""

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
            outputs = self(images.view(images.shape[0], -1))
            # Obtain predictons
            _, preds = torch.max(outputs.data, dim=1)
            # Store predictions
            all_preds = torch.cat((all_preds, preds), 0)
        # Return predictions
        return all_preds
