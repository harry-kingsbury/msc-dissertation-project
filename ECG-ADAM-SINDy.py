"""ECG-ADAM-SINDy: Sparse Identification of Nonlinear Dynamics using ADAM Optimization

This module applies ADAM-SINDy to ECG data by:
1. Loading ECG records and extracting RR intervals (heart rate variability)
2. Building a polynomial library of features (1, RR, RR^2, etc.)
3. Using PyTorch's ADAM optimizer with L1 regularization to discover sparse dynamics
4. Simulating the discovered equations and evaluating accuracy
"""

import numpy as np
import os
import torch  # PyTorch for neural network operations
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms (ADAM)
from sklearn.metrics import mean_squared_error  # MSE metric for evaluation
import wfdb  # WFDB toolkit for loading cardiac signal data
from wfdb import processing  # Signal processing utilities (R-peak detection)


def adam_sindytest(simulated_data, actual_data, label="Heart Condition",
                   qt_min=None, qt_max=None, rr_mean=None, rr_std=None):
    """Evaluate ADAM-SINDy model performance by comparing simulated vs actual cardiac data.
    
    Calculates Mean Squared Error (MSE) for QT and RR intervals in both scaled and
    physical units. Determines system stability based on error thresholds.
    
    Args:
        simulated_data: Model predictions (n_samples, 2) - [QT, RR]
        actual_data: Ground truth data (n_samples, 2) - [QT, RR]
        label: Description of the cardiac condition
        rr_mean, rr_std: Scaling parameters to convert from standardized to physical units
    """
    print(f"\n{'='*50}\nVALIDATING ADAM-SINDY MODEL: {label}\n{'='*50}")

    # Compute MSE between predictions and ground truth
    qt_mse_scaled = mean_squared_error(actual_data[:, 0], simulated_data[:, 0])
    rr_mse_scaled = mean_squared_error(actual_data[:, 1], simulated_data[:, 1])

    print("ADAM-SINDy Predictive Accuracy Metrics:")
    print(f"   |-- QT Scaled MSE: {qt_mse_scaled:.8f}")
    print(f"   |-- RR Scaled MSE: {rr_mse_scaled:.8f}")

    # Convert predictions back to physical units (milliseconds) if scaling parameters provided
    if rr_mean is not None and rr_std is not None:
        pred_rr = simulated_data[:, 1] * rr_std + rr_mean
        true_rr = actual_data[:, 1] * rr_std + rr_mean
        rr_error = mean_squared_error(true_rr, pred_rr)
        print(f"   |-- RR Interval Equation Error Rate: {rr_error:.8f} ms^2")

    # Assess system stability: lower MSE indicates better model fit
    if rr_mse_scaled < 0.5:
        print("Stable System: The ADAM-SINDy equations successfully captured the cardiac rhythm.")
    else:
        print("High Deviation: The model equations are drifting from the true heart timeline.")
    print("="*50 + "\n")


class AdamSINDyModel(nn.Module):
    """Neural network model for discovering sparse nonlinear dynamics via ADAM optimization.
    
    Implements a library of polynomial features and learns sparse coefficients that
    govern the dynamics: x[k+1] = Library(x[k]) @ Coefficients
    """
    def __init__(self, n_features=2):
        super(AdamSINDyModel, self).__init__()
        self.n_features = n_features  # Number of state variables (QT, RR)
        # Library basis: [1, RR, RR^2] - monomials up to degree 2
        self.n_library = 6
        # Learnable coefficients: shape (n_features, n_library) for linear regression
        self.coefficients = nn.Parameter(torch.randn(n_features, self.n_library) * 0.01)

    def build_library(self, X):
        """Construct polynomial feature library from state variables.
        
        Creates basis functions: [1, QT, RR, QT^2, RR^2, QT*RR]
        These are the candidate terms from which sparse dynamics emerge.
        """
        x = X[:, 0:1]  # QT interval
        y = X[:, 1:2]  # RR interval (heart rate variability)
        ones = torch.ones_like(x)  # Bias term
        # Concatenate polynomial terms: constant, linear, and quadratic combinations
        lib = torch.cat([ones, x, y, x**2, y**2, x * y], dim=1)
        return lib

    def forward(self, X):
        """Predict next state using discovered sparse equations.
        
        Returns: X[k+1] = Library(X[k]) @ Coefficients^T
        """
        lib = self.build_library(X)  # Build feature basis
        # Linear combination: each feature multiplied by its learned coefficient
        return torch.matmul(lib, self.coefficients.t())

    def fit_adam(self, X_data, epochs=2000, lr=0.01, l1_lambda=0.001):
        """Train model using ADAM optimizer with L1 regularization for sparsity.
        
        Args:
            X_data: Training trajectory (n_samples, n_features)
            epochs: Number of training iterations
            lr: Learning rate for ADAM optimizer
            l1_lambda: L1 regularization weight to encourage sparse coefficients
        
        The loss combines MSE (prediction accuracy) + L1 (coefficient sparsity).
        This forces small coefficients to zero, discovering only essential terms.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)  # ADAM: adaptive learning rate optimizer
        # Prepare training data: predict X[k+1] from X[k]
        X_train = torch.tensor(X_data[:-1], dtype=torch.float32)  # Current states
        X_target = torch.tensor(X_data[1:], dtype=torch.float32)  # Next states (targets)

        self.train()  # Set to training mode
        for epoch in range(epochs):
            optimizer.zero_grad()  # Reset gradients
            prediction = self(X_train)  # Forward pass: predict next states
            # MSE loss: prediction accuracy
            mse_loss = torch.mean((prediction - X_target) ** 2)
            # L1 loss: penalize large coefficient values to encourage sparsity
            l1_loss = l1_lambda * torch.sum(torch.abs(self.coefficients))
            # Total loss = accuracy + sparsity
            loss = mse_loss + l1_loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

    def simulate(self, initial_state, steps):
        """Simulate the discovered dynamics forward in time.
        
        Starting from initial_state, iteratively apply the learned equations
        to generate a trajectory of length 'steps'.
        
        Args:
            initial_state: Starting condition (n_features,)
            steps: Number of time steps to simulate
        
        Returns: Simulated trajectory (steps, n_features)
        """
        self.eval()  # Set to evaluation mode (no dropout, fixed batch norm)
        simulated = np.zeros((steps, self.n_features))  # Allocate output array
        simulated[0] = initial_state  # Initialize with starting state
        curr_tensor = torch.tensor(initial_state.reshape(1, 2), dtype=torch.float32)
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            for k in range(1, steps):
                try:
                    # Apply learned dynamics to current state
                    next_tensor = self(curr_tensor)
                    simulated[k] = next_tensor.numpy().flatten()
                    curr_tensor = next_tensor  # Continue from new state
                except Exception:
                    # Handle numerical errors (e.g., NaN) by holding last state
                    simulated[k] = simulated[k-1]
        return simulated

    def print_equations(self):
        """Display discovered equations in human-readable format.
        
        Prints the sparse coefficients, showing only terms that exceed
        a threshold (1e-3) to focus on significant dynamics.
        """
        feature_names = ['1', 'QT', 'RR', 'QT^2', 'RR^2', 'QT*RR']  # Library basis labels
        target_names = ['QT[k+1]', 'RR[k+1]']  # Output variables
        coefs = self.coefficients.detach().numpy()  # Extract learned coefficients

        print("\n--- ADAM-SINDy Discovered Equations ---")
        for i, target in enumerate(target_names):
            terms = []
            for j, name in enumerate(feature_names):
                val = coefs[i, j]
                # Include only significant terms (above threshold)
                if abs(val) > 1e-3:
                    sign = "+" if val >= 0 else "-"  # Format as +/- for clarity
                    terms.append(f"{sign} {abs(val):.4f} * {name}")
            # Display equation or '0' if no significant terms
            if not terms:
                eq_str = "0"
            else:
                eq_str = " ".join(terms)
            print(f"{target} = {eq_str}")
        print("-" * 42)


def load_ecg_subset_for_adam(record_name):
    """Load ECG data from WFDB format and extract heart rate variability features.
    
    Processes raw ECG signal to:
    1. Detect R-peaks (ventricular activations)
    2. Calculate RR intervals (time between consecutive R-peaks)
    3. Standardize RR intervals for modeling
    4. Remove outliers and physiologically implausible values
    
    Args:
        record_name: Base name of WFDB record (without file extension)
    
    Returns:
        X: Feature matrix (n_samples, 2) - [QT_scaled, RR_scaled]
        qt_min, qt_max: QT interval bounds (placeholder zeros)
        rr_mean, rr_std: Normalization parameters for RR intervals
    """
    # Verify WFDB files exist
    if not os.path.exists(record_name + ".dat") or not os.path.exists(record_name + ".hea"):
        raise FileNotFoundError(f"WFDB files for record '{record_name}' not found.")

    # Load ECG signal
    rec = wfdb.rdrecord(record_name)
    ecg = rec.p_signal[:, 0]  # First channel (typically Lead II)
    fs = rec.fs  # Sampling frequency (Hz)

    # Detect R-peaks using Goulart-Quezada algorithm
    r_peaks = processing.gqrs_detect(ecg, fs)

    # Calculate RR intervals (time between R-peaks)
    rr_samples = np.diff(r_peaks)  # Number of samples between peaks
    rr_ms = (rr_samples / fs) * 1000.0  # Convert to milliseconds
    rr_ms = rr_ms[np.isfinite(rr_ms)]  # Remove NaN/inf values

    # Outlier removal: keep physiologically plausible RR intervals
    # (typically 300-2000 ms corresponds to heart rates 30-200 bpm)
    mask = (rr_ms > 300) & (rr_ms < 2000)
    rr_ms = rr_ms[mask]

    # Standardize RR intervals for model training (zero mean, unit variance)
    rr_mean = rr_ms.mean()
    rr_std = rr_ms.std()
    rr_scaled = (rr_ms - rr_mean) / (rr_std + 1e-8)

    # QT interval (QRS-to-T wave duration) - using placeholder zeros for now
    qt_scaled = np.zeros_like(rr_scaled)
    qt_min, qt_max = 0.0, 0.0

    # Stack features: each row is [QT[k], RR[k]]
    X = np.vstack((qt_scaled, rr_scaled)).T
    return X, qt_min, qt_max, rr_mean, rr_std


def run_adam_sindy_pipeline(record_name, label):
    """Execute complete ADAM-SINDy pipeline: load data, train model, forecast, evaluate.
    
    Args:
        record_name: WFDB record base name
        label: Description for output messages
    """
    # Load and preprocess ECG data
    X, qt_min, qt_max, rr_mean, rr_std = load_ecg_subset_for_adam(record_name)
    print(f"\nLoaded {len(X)} points mapping: {label}")

    # Train/test split (70% training, 30% testing)
    split_idx = int(len(X) * 0.7)
    split_idx = max(5, min(split_idx, len(X) - 1))  # Ensure reasonable split
    X_train = X[:split_idx]  # Training data
    X_test = X[split_idx:]   # Test data (unseen during training)

    print(f"Training on {len(X_train)} steps | Testing on {len(X_test)} steps")

    # Initialize and train ADAM-SINDy model
    model = AdamSINDyModel(n_features=2)
    model.fit_adam(X_train, epochs=2500, lr=0.005, l1_lambda=0.0005)

    # Display discovered equations
    model.print_equations()

    # Forecast test segment starting from last training point
    simulated_test = model.simulate(X_train[-1], len(X_test))

    # Evaluate model performance on test data
    adam_sindytest(
        simulated_test,
        X_test,
        label=label,
        qt_min=qt_min,
        qt_max=qt_max,
        rr_mean=rr_mean,
        rr_std=rr_std
    )


def main():
    """Main entry point: run ADAM-SINDy on configured ECG records."""
    # Configuration: list of (record_name, description) pairs to analyze
    configs = [
        ('ARR_02', "Subject ARR_02 ECG (Adam-SINDy)"),
    ]

    # Process each record
    for record_name, label in configs:
        run_adam_sindy_pipeline(record_name, label)


if __name__ == "__main__":
    main()