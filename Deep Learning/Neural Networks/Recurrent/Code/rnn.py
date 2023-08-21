"""
This file creates a Recurrent Neural Network (RNN).
This RNN has been trained and tested on the Mastercard Stock dataset.
This means that the for other datasets, the RNN architecture and output prediction conversion may need alterting.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Callable

# Third Party
import torch
import torch.nn as nn

# Private

# -------------------------------------------------------------------------------------------------------------------- #

# Recurrent Neural Network


class RNN(nn.Module):
    """This initialises the RNN."""

    class RNNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(RNNModel, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim
            # Layer dimensions
            self.layer_dim = layer_dim
            # RNN
            self.rnn = nn.RNN(
                input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity="relu"
            )
            # Output layer
            self.fc = nn.Linear(hidden_dim, output_dim)

    """This applies a forward pass to the input data."""

    def forward(self, x):
        # Initial hidden state
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # Backpropagation Through Time (detach the hidden state)
        out, _ = self.rnn(x, h_0.detach())
        # Output (Shape: [batch_size, seq_length, hidden_size])
        out = self.fc(out[:, -1, :])
        return out

    """This fits the RNN to the training set and evaluates the RNN on the validation set."""

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
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # Initialise training loss
            train_loss = 0
            # Iterate through entire collection (per batch)
            for seq, targets in train_loader:
                # Clear gradients with respect to parameters
                opt.zero_grad()
                # Forward pass
                outputs = self(seq)
                # Calculate loss
                loss = loss_fn(outputs, targets)
                # Accumulate loss
                train_loss += loss.item()
                # Backward pass
                loss.backward()
                # Parameter update
                opt.step()
            # Store information
            train_losses.append(train_loss)
            # Evaluate model (NOTE: In the evaluation phase, the model parameters do not need updating!)
            with torch.no_grad():
                # Initialise validation loss
                val_loss = 0
                # Iterate through entire collection (per batch)for inputs, labels in test_loader:
                for seq, targets in val_loader:
                    # Forward pass
                    outputs = self(seq)
                    # Calculate loss
                    loss = loss_fn(outputs, targets)
                    # Accumulate loss
                    val_loss += loss.item()
                # Store information
                val_losses.append(val_loss)
                # Print loss and accuracy
                print(
                    f"Epoch: {epoch} | Validation loss: {val_loss} | Validation MSE: {val_loss}"
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
                        early_stop = True
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
        for seq, _ in test_loader:
            # Forward pass
            outputs = self(seq)
            # Obtain predictons
            _, preds = torch.max(outputs.data, dim=1)
            # Store predictions
            all_preds = torch.cat((all_preds, preds), 0)
        # Return predictions
        return all_preds
