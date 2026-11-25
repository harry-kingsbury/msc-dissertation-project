"""ECG-DMD: Dynamic Mode Decomposition for Cardiac Forecasting

This module applies DMD (Dynamic Mode Decomposition) to ECG data by:
1. Loading ECG records and extracting RR intervals (heart rate variability)
2. Building a data matrix where each column is a time snapshot
3. Computing DMD modes, eigenvalues, and amplitudes via SVD
4. Forecasting future dynamics and evaluating accuracy

DMD discovers low-rank linear dynamics that govern the signal evolution.
"""

import numpy as np  # Numerical computing
import os  # File operations
from pydmd import DMD  # Dynamic Mode Decomposition
from sklearn.metrics import mean_squared_error  # MSE metric for evaluation
import wfdb  # WFDB toolkit for cardiac signal data
from wfdb import processing  # R-peak detection


def load_ecg_rr_only(record_name):
    """Load ECG and extract RR intervals (time between R-peaks).
    
    Args:
        record_name: Base name of WFDB record
    
    Returns:
        rr_scaled: Standardized RR intervals
        rr_mean, rr_std: Normalization parameters
    """
    # Verify WFDB files exist
    if not os.path.exists(record_name + ".dat") or not os.path.exists(record_name + ".hea"):
        raise FileNotFoundError(f"WFDB files for record '{record_name}' not found.")

    # Load ECG signal
    rec = wfdb.rdrecord(record_name)
    ecg = rec.p_signal[:, 0]  # First channel
    fs = rec.fs  # Sampling frequency

    # Detect R-peaks
    r_peaks = processing.gqrs_detect(ecg, fs)

    # Compute RR intervals in milliseconds
    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs) * 1000.0
    rr_ms = rr_ms[np.isfinite(rr_ms)]  # Remove NaN/inf

    # Remove outliers: keep physiologically plausible heart rates (300-2000 ms)
    mask = (rr_ms > 300) & (rr_ms < 2000)
    rr_ms = rr_ms[mask]

    # Standardize: zero mean, unit variance for numerical stability
    rr_mean = rr_ms.mean()
    rr_std = rr_ms.std()
    rr_scaled = (rr_ms - rr_mean) / (rr_std + 1e-8)

    return rr_scaled, rr_mean, rr_std


def forecast_dmd_1d(dmd_model, x0, n_test):
    """Generate forecasts using trained DMD model.
    
    Reconstructs the signal trajectory from DMD modes and eigenvalues:
    x[k] = sum_j(phi_j * lambda_j^k * amplitude_j)
    
    Args:
        dmd_model: Trained DMD model with modes and eigenvalues
        x0: Initial condition (starting value)
        n_test: Number of steps to forecast
    
    Returns: Forecasted trajectory (n_test,)
    """
    # Check if DMD model was successfully computed
    if not hasattr(dmd_model, "modes") or not hasattr(dmd_model, "eigs"):
        # Fallback: return constant if decomposition failed
        return np.full(n_test, x0)

    # Extract DMD components
    Phi = np.asarray(dmd_model.modes)   # Mode vectors (spatial structure)
    eigs = np.asarray(dmd_model.eigs)   # Eigenvalues (time dynamics)

    # Compute initial amplitudes by solving: Phi * z = x0
    x0_arr = np.array([[x0]])
    z = np.linalg.pinv(Phi) @ x0_arr  # Pseudo-inverse gives least-squares solution

    # Iterate forward in time
    forecast = []
    for k in range(n_test):
        # Apply exponential growth: z[k+1] = Lambda * z[k]
        z = eigs[:, None] * z
        # Reconstruct signal: x[k] = Phi @ z[k]
        x_next = (Phi @ z).real  # Take real part (may have small imaginary noise)
        forecast.append(x_next[0, 0])

    return np.array(forecast)


def dmd_test_1d(dmd_model, train_rr, test_rr, label="Heart Condition",
                rr_mean=None, rr_std=None):
    """Evaluate DMD forecasting performance.
    
    Compares predicted vs actual RR intervals on held-out test data.
    Reports MSE in both scaled and physical units.
    
    Args:
        dmd_model: Trained DMD model
        train_rr: Training RR intervals (used to initialize forecast)
        test_rr: Ground truth test RR intervals
        label: Description for output
        rr_mean, rr_std: Scaling parameters to convert back to physical units
    """
    print(f"\n{'='*50}\nVALIDATING DMD FORECASTING MODEL: {label}\n{'='*50}")

    # Forecast from last training point
    x0 = train_rr[-1]  # Starting from last observed RR
    forecasted = forecast_dmd_1d(dmd_model, x0, len(test_rr))

    # Verify shapes match
    if forecasted.shape != test_rr.shape:
        raise ValueError(f"Shape mismatch: forecast {forecasted.shape}, test {test_rr.shape}")

    # Compute MSE in scaled units
    rr_mse_scaled = mean_squared_error(test_rr, forecasted)

    print("DMD Out-of-Sample Forecasting Accuracy Metrics:")
    print(f"   |-- RR Scaled MSE: {rr_mse_scaled:.8f}")

    # Convert back to physical units (milliseconds) if scaling info provided
    if rr_mean is not None and rr_std is not None:
        pred_rr = forecasted * rr_std + rr_mean
        true_rr = test_rr * rr_std + rr_mean
        rr_error = mean_squared_error(true_rr, pred_rr)
        print(f"   |-- RR Interval Forecast Error: {rr_error:.8f} ms^2")

    # Assess stability
    if rr_mse_scaled < 0.5:
        print("Stable System: The DMD model successfully forecasted future cardiac dynamics.")
    else:
        print("High Deviation: The DMD forecast drifted from the true test segment.")
    print("="*50 + "\n")


def run_dmd_analysis(record_name, label):
    """Execute complete DMD pipeline: load data, compute decomposition, forecast, evaluate.
    
    Args:
        record_name: WFDB record base name
        label: Description for output messages
    """
    # Load and preprocess
    rr_scaled, rr_mean, rr_std = load_ecg_rr_only(record_name)
    print(f"\n--- DMD FORECASTING RESULTS: {label} ---")
    print(f"Loaded {len(rr_scaled)} RR beats")

    # Train/test split (70/30)
    split_idx = int(len(rr_scaled) * 0.7)
    split_idx = max(2, min(split_idx, len(rr_scaled) - 1))

    X_train = rr_scaled[:split_idx]
    X_test = rr_scaled[split_idx:]

    print(f"Training on {len(X_train)} steps | Testing on {len(X_test)} steps")

    # Fit DMD: requires data in shape (1, n_time_steps) for 1D time series
    dmd = DMD(svd_rank=2)  # Use 2 modes (adjustable based on complexity)
    dmd.fit(X_train[None, :])  # Reshape to (1, n_train) for DMD

    # Display discovered eigenvalues (growth/decay rates)
    print(f"DMD Eigenvalues: {dmd.eigs}")

    # Evaluate on test set
    dmd_test_1d(
        dmd,
        X_train,
        X_test,
        label=label,
        rr_mean=rr_mean,
        rr_std=rr_std
    )


def main():
    """Main entry point: run DMD on configured ECG records."""
    configs = [
        ('ARR_02', "Subject ARR_02 ECG (DMD)"),
    ]

    for record_name, label in configs:
        run_dmd_analysis(record_name, label)


if __name__ == "__main__":
    main()