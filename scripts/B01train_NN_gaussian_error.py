import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from matplotlib import pyplot as plt
import time
from torchsummary import summary
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_examples import TinyCNN
from helper import normalize, denormalize, denormalize_std, train_model, get_normalized_data, evaluate_model

DATA_PATH = "/Users/nilhe916/PhD_Uppsala/2026/teaching/test_files"

# Hyperparameters
learning_rate = 0.8e-5
batch_size = 32
num_epochs =3
patience = 20 # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7 # Fraction of the data used for training
val_fraction = 0.15 # Fraction of the data used for validation

# Call the function to get normalized data
spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)

# Convert numpy arrays to PyTorch tensors
spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Split the data into training, validation, and test sets
total_samples = len(spectra_tensor)
train_size = int(train_fraction * total_samples)
val_size = int(val_fraction * total_samples)
test_size = total_samples - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(TensorDataset(spectra_tensor, labels_tensor), [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Create DataLoaders for batching and shuffling the data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_name = "CNN_1_gaussian_error"
class TinyCNN(nn.Module):
    def __init__(self, nLabels):
        super(TinyCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            # nn.AvgPool1d(1),

            nn.Conv1d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.AvgPool1d(3),

            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.AvgPool1d(3),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Conv1d(128, 128, kernel_size=1),
            nn.Dropout(0.2),

            nn.Linear(907, 32), # batch, filters, * -> batch, filters, 32
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*32, 128),
            nn.ReLU(),
            nn.Linear(128, nLabels)
        )

    def forward(self, x):
        # x = torch.unsqueeze(x,1)
        x = self.model(x)
        return x


# Define the negative log-likelihood (NLL) loss function.
# This loss function is appropriate for regression tasks where we predict both values and uncertainties.
def nll_loss(inputs, batch_labels, model):
    """
    Calculate the negative log-likelihood (NLL) loss.

    Parameters
    ----------
    inputs : torch.Tensor
        The input tensor to the model.
    batch_labels : torch.Tensor
        The ground truth labels for the batch.
    model : nn.Module
        The neural network model.

    Returns
    -------
    torch.Tensor
        The calculated NLL loss.
    """

    predictions=model(inputs)

    mean = predictions[:, :n_labels]  # Extract the mean values
    log_std = predictions[:, n_labels:]  # Extract the log standard deviations
    std = torch.exp(log_std)  # Convert log standard deviation to standard deviation
    return torch.mean((0.5 * ((batch_labels - mean) / std) ** 2) + log_std)  # NLL formula

loss_function = nll_loss


if __name__ == "__main__":

    model = TinyCNN(n_labels * 2) # times two beacuse we want to predict the mean and the standard deviation
    # Print the model summary before moving it to the device
    summary(model, input_size=(1, spectra_length))

    # Detect and use Apple Silicon GPU (MPS) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)


    # Call the function
    train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience, device)

    # Final evaluation on the test dataset
    model.load_state_dict(best_model)  # Load the best model
    # Save the best model to the "models" directory
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(best_model, f'models/{model_name}_best.pth')
    model.to(device)
    all_predictions, all_true_labels,_,_ = evaluate_model(model, test_loader, loss_function, device)
    pred_mean = denormalize(all_predictions[:, :n_labels], ranges)  # Denormalize predictions
    pred_std = denormalize_std(np.exp(all_predictions[:, n_labels:]), ranges)  # Extract the predicted standard deviations
    all_true_labels = denormalize(all_true_labels, ranges)  # Denormalize true labels

    # Check if the "plots" directory exists, if not, create it
    if not os.path.exists('plots/%s' % model_name):
        os.makedirs('plots/%s' % model_name)
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    # plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/training_validation_loss.png')

    # Scatter plots for predictions
    plt.figure(figsize=(16,7.5))
    for j in range(n_labels):
        plt.subplot(1,3,j+1)
        gt = all_true_labels
        plt.scatter(gt[:,j],y=pred_mean[:,j],s=6,alpha=0.2)
        plt.plot([gt[:,j].min().item(),gt[:,j].max().item()],[gt[:,j].min().item(),gt[:,j].max().item()],c="black",linestyle="dashed",label="Perfect prediction")
        plt.xlabel("true "+labelNames[j])
        plt.ylabel("predicted "+labelNames[j])
        plt.legend()
    plt.tight_layout()
    plt.savefig('plots/scatter.png')

    # Plot pull distributions for the three labels
    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        pull = (gt - pred) / std  # Calculate the pull
        plt.hist(pull, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
        plt.xlabel(f'Pull for {labelNames[j]}')
        plt.ylabel('Frequency')
        plt.title(f'Pull Distribution for {labelNames[j]}')
        plt.axvline(pull.mean(), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(pull.std(), color='green', linestyle='dashed', linewidth=1)
        plt.text(0.95, 0.95, f'Mean: {pull.mean():.2f}\nStd: {pull.std():.2f}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig('plots/pull.png')

    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        diff = gt - pred
        plt.hist(diff, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
        plt.xlabel(f'True - Predicted {labelNames[j]}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution for {labelNames[j]}')
        plt.axvline(diff.mean(), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(diff.std(), color='green', linestyle='dashed', linewidth=1)
        plt.text(0.95, 0.95, f'Mean: {diff.mean():.2f}\nStd: {diff.std():.2f}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig('plots/true_predicted.png')

    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        plt.hist(std, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
        plt.xlabel(f'STD for {labelNames[j]}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of STD for {labelNames[j]}')
        plt.axvline(std.mean(), color='red', linestyle='dashed', linewidth=1)
        plt.axvline(std.std(), color='green', linestyle='dashed', linewidth=1)
        plt.text(0.95, 0.95, f'Mean: {std.mean():.2f}\nStd: {std.std():.2f}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig('plots/std.png')
    plt.show()
