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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from model_examples import TinyCNN
from helper import normalize, denormalize, denormalize_std, train_model, get_normalized_data, evaluate_model
from B01train_NN_gaussian_error import TinyCNN, test_loader, loss_function, n_labels, spectra_length, ranges, labelNames

model_name = "CNN_1_gaussian_error"

# Detect and use Apple Silicon GPU (MPS) if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Load the model from the .pth file
model = TinyCNN(n_labels * 2)
model.load_state_dict(torch.load(f'models/{model_name}_best.pth', map_location=device, weights_only=True))


model.to(device)
# Print the model summary before moving it to the device
# summary(model, input_size=(1, spectra_length))

all_predictions, all_true_labels,_,_ = evaluate_model(model, test_loader, loss_function, device)
pred_mean = denormalize(all_predictions[:, :n_labels], ranges)  # Denormalize predictions
pred_std = denormalize_std(np.exp(all_predictions[:, n_labels:]), ranges)  # Extract the predicted standard deviations
all_true_labels = denormalize(all_true_labels, ranges)  # Denormalize true labels

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
