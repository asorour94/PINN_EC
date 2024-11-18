# -*- coding: utf-8 -*-
"""PINN_EC(18/11).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wNoQqzos3QPpDg96wnIT2KFXFI9pj2lu

torch is used for implementing the PINN model.
sklearn handles preprocessing, splitting, and metrics.
Random seeds ensure consistent results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Set a random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

"""Features and targets are selected based on the identified variables.
Data normalization scales values between 0 and 1 for better training convergence.
Data is split into training and validation sets and converted to PyTorch tensors
"""

# Load the dataset
data = pd.read_csv("Data-Melbourne_F_fixed.csv")

# Feature selection
features = ["Average Outflow", "Average Inflow", "Ammonia", "Biological Oxygen Demand", "Chemical Oxygen Demand",
            "Total Nitrogen", "Average Temperature", "Average humidity", "Total rainfall"]
target = "Energy Consumption"

X = data[features].values
y = data[target].values.reshape(-1, 1)

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

"""The model has two hidden layers with 12 neurons each, using a sigmoid activation function.
The output layer provides a single value for energy consumption
"""

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden1 = nn.Linear(9, 12)
        self.hidden2 = nn.Linear(12, 12)
        self.output_layer = nn.Linear(12, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.output_layer(x)

"""Physics loss calculates residuals of the simplified ODE.
Combined loss combines the MSE for data fit and the physics-informed loss.
"""

def physics_loss(inputs, outputs, alpha, beta, gamma, T_ref):
    Q_in, Q_out, ammonia, BOD, COD, total_nitrogen, T_avg, RH, rainfall = torch.split(inputs, 1, dim=1)

    # Simplified ODE terms
    delta_DO = COD + BOD - ammonia  # Proxy for oxygen demand
    predicted_energy = outputs
    residual = predicted_energy - (alpha * Q_in * delta_DO + beta * Q_out + gamma * (T_avg - T_ref))
    return torch.mean(residual ** 2)

def combined_loss(model, inputs, targets, alpha, beta, gamma, T_ref):
    outputs = model(inputs)
    mse_loss = torch.mean((targets - outputs) ** 2)
    phys_loss = physics_loss(inputs, outputs, alpha, beta, gamma, T_ref)
    return mse_loss + phys_loss

"""The model is trained using a combined loss function.
Batches improve computational efficiency.
Validation loss tracks performance on unseen data
"""

# Hyperparameters
alpha, beta, gamma = 0.01, 0.005, 0.002
T_ref = torch.tensor(15.0, dtype=torch.float32)  # Reference temperature
learning_rate = 0.001
epochs = 1500
batch_size = 32

# Initialize model, optimizer, and loss function
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loaders for batch processing
train_loader = torch.utils.data.DataLoader(
    list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False
)

# Training loop
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = combined_loss(model, batch_X, batch_y, alpha, beta, gamma, T_ref)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            loss = combined_loss(model, batch_X, batch_y, alpha, beta, gamma, T_ref)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

"""Evaluate model performance using MSE, RMSE, and
𝑅
2
R
2
  metrics.
Predictions are transformed back to their original scale for interpretation
"""

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train).numpy()
    y_pred_val = model(X_val).numpy()

y_train_actual = scaler_y.inverse_transform(y_train.numpy())
y_val_actual = scaler_y.inverse_transform(y_val.numpy())
y_pred_train_actual = scaler_y.inverse_transform(y_pred_train)
y_pred_val_actual = scaler_y.inverse_transform(y_pred_val)

mse = mean_squared_error(y_val_actual, y_pred_val_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_val_actual, y_pred_val_actual)

print(f"Validation MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

"""Loss plots help assess convergence during training.
Scatter plot visualizes the relationship between actual and predicted energy consumption.
"""

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# Plot predicted vs actual energy consumption
plt.figure(figsize=(10, 6))

# Actual values in red
plt.scatter(range(len(y_val_actual)), y_val_actual, color="red", label="Actual", alpha=0.7)

# Predicted values in green
plt.scatter(range(len(y_pred_val_actual)), y_pred_val_actual, color="green", label="Predicted", alpha=0.7)

plt.xlabel("Sample Index")
plt.ylabel("Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")
plt.legend()
plt.show()