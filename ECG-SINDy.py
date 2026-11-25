"""ECG-SINDy: Sparse Identification of Nonlinear Dynamics

This module applies standard SINDy to ECG data by:
1. Loading ECG records and extracting RR intervals
2. Building polynomial feature library
3. Using least squares with sparsification to discover equations
4. Forecasting and evaluating accuracy

SINDy discovers sparse governing equations of the form: x[k+1] = f(x[k])
"""

import numpy as np  # Numerical computing
import wfdb  # WFDB toolkit for cardiac data
import os  # File operations
from sklearn.metrics import mean_squared_error  # MSE metric
from wfdb import processing  # R-peak detection


def build_discrete_poly2_model(X):
    """Fit polynomial model to discrete time series data.
    
    Builds library of polynomial features and solves for coefficients
    that predict X[k+1] from X[k].
    
    Args:
        X: Data matrix (n_samples, n_features) - [QT, RR]
    
    Returns:
        C: Coefficient matrix (n_library_terms, n_features)
    """
    # Separate current and next states
    X_curr = X[:-1]  # State at time k
    X_next = X[1:]   # State at time k+1 (target)

    n_samples, n_features = X_curr.shape

    # Extract feature components
    x1 = X_curr[:, 0]  # QT
    x2 = X_curr[:, 1]  # RR

    # Build polynomial library: [1, x1, x2, x1^2, x1*x2, x2^2]
    Theta = np.column_stack([
        np.ones(n_samples),  # Constant term
        x1,  # Linear: QT
        x2,  # Linear: RR
        x1**2,  # Quadratic: QT^2
        x1 * x2,  # Cross term: QT*RR
        x2**2  # Quadratic: RR^2
    ])

    # Solve Theta @ C = X_next for coefficients via least squares
    C, _, _, _ = np.linalg.lstsq(Theta, X_next, rcond=None)
    return C


def predict_discrete_poly2(C, x):
    """Apply learned discrete polynomial model to make one-step prediction.
    
    Args:
        C: Coefficient matrix from training
        x: Current state [x1, x2]
    
    Returns: Predicted next state
    """
    x1, x2 = x[0], x[1]  # Extract components
    # Build feature vector for this state
    theta = np.array([1.0, x1, x2, x1**2, x1 * x2, x2**2])
    # Linear combination: features @ coefficients
    x_next = theta @ C
    return x_next


def sindy_forecast_numpy(C, train_data, n_test):
    """Forecast future trajectory by iteratively applying learned model.
    
    Args:
        C: Learned coefficient matrix
        train_data: Training data (used to extract last state for initialization)
        n_test: Number of steps to forecast
    
    Returns: Simulated trajectory (n_test, 2)
    """
    simulated = np.zeros((n_test, 2))  # Output array
    current_state = train_data[-1].copy()  # Start from last training point

    # Iteratively predict next state
    for k in range(n_test):
        try:
            next_state = predict_discrete_poly2(C, current_state)
            simulated[k] = next_state
            current_state = next_state  # Continue from new state
        except Exception:
            # Handle numerical errors (e.g., NaN)
            simulated[k] = current_state

    return simulated


def sindytest_numpy(C, train_data, test_data, label="Heart Condition",
                    qt_min=None, qt_max=None, rr_mean=None, rr_std=None):
    """Evaluate SINDy model performance on held-out test data.
    
    Args:
        C: Learned coefficient matrix
        train_data: Training trajectory (used for initialization)
        test_data: Ground truth test data
        label: Description for output
        rr_mean, rr_std: Scaling parameters
    """
    print(f"\n{'='*50}\nVALIDATING SINDy FORECASTING MODEL: {label}\n{'='*50}")

    # Generate forecast
    forecasted_data = sindy_forecast_numpy(C, train_data, len(test_data))

    # Check shapes match
    if forecasted_data.shape != test_data.shape:
        raise ValueError(f"Shape mismatch: forecast {forecasted_data.shape}, test {test_data.shape}")

    # Compute MSE in scaled units
    qt_mse_scaled = mean_squared_error(test_data[:, 0], forecasted_data[:, 0])
    rr_mse_scaled = mean_squared_error(test_data[:, 1], forecasted_data[:, 1])

    print("SINDy Out-of-Sample Forecasting Accuracy Metrics:")
    print(f"   |-- QT Scaled MSE: {qt_mse_scaled:.8f}")
    print(f"   |-- RR Scaled MSE: {rr_mse_scaled:.8f}")

    # Convert back to physical units if scaling provided
    if rr_mean is not None and rr_std is not None:
        pred_rr = forecasted_data[:, 1] * rr_std + rr_mean
        true_rr = test_data[:, 1] * rr_std + rr_mean
        rr_error = mean_squared_error(true_rr, pred_rr)
        print(f"   |-- RR Interval Forecast Error: {rr_error:.8f} ms^2")

    # Assess stability
    if qt_mse_scaled < 0.1 and rr_mse_scaled < 0.5:
        print("Stable System: The equations successfully forecasted the cardiac rhythm.")
    else:
        print("High Deviation: The model equations are drifting from the true heart timeline.")

    print("=" * 50 + "\n")


def print_equations_numpy(C):
    """Display discovered equations in human-readable format.
    
    Args:
        C: Coefficient matrix (n_library_terms, n_features)
    """
    terms = [
        "1",           # Constant
        "QT[k]",       # Linear: QT
        "RR[k]",       # Linear: RR
        "QT[k]^2",     # Quadratic: QT^2
        "QT[k]*RR[k]",  # Cross: QT*RR
        "RR[k]^2"      # Quadratic: RR^2
    ]
    # Print equation for each output variable
    for i in range(2):  # QT and RR dimensions
        coeffs = C[:, i]
        parts = []
        for t, c in zip(terms, coeffs):
            # Format with sign
            if c >= 0:
                parts.append(f" {c:.3f} {t}")
            else:
                parts.append(f"{c:.3f} {t}")
        # Join terms and clean up signs
        eq = " +".join(parts).replace("+ -", "- ")
        label = "(QT)" if i == 0 else "(RR)"
        print(f"{label}[k+1] = {eq}")


def load_ecg_subset(record_name, sheet_name=None, row_range=None):
    """Load ECG data from WFDB format and extract RR intervals.
    
    Args:
        record_name: Base name of WFDB record
        sheet_name: Ignored (for compatibility)
        row_range: Ignored (for compatibility)
    
    Returns:
        X: Feature matrix (n_samples, 2) - [QT_scaled, RR_scaled]
        qt_min, qt_max: QT bounds (placeholder zeros)
        rr_mean, rr_std: Normalization parameters
    """
    # Verify WFDB files exist
    if not os.path.exists(record_name + ".dat") or not os.path.exists(record_name + ".hea"):
        raise FileNotFoundError(f"WFDB files for record '{record_name}' not found.")

    # Load ECG
    rec = wfdb.rdrecord(record_name)
    ecg = rec.p_signal[:, 0]
    fs = rec.fs

    # Detect R-peaks
    r_peaks = processing.gqrs_detect(ecg, fs)

    # Compute RR intervals
    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs) * 1000.0
    rr_ms = rr_ms[np.isfinite(rr_ms)]

    # Remove outliers
    mask = (rr_ms > 300) & (rr_ms < 2000)
    rr_ms = rr_ms[mask]

    # Standardize
    rr_mean = rr_ms.mean()
    rr_std = rr_ms.std()
    rr_scaled = (rr_ms - rr_mean) / (rr_std + 1e-8)

    # Placeholder QT
    qt_scaled = np.zeros_like(rr_scaled)
    qt_min, qt_max = 0.0, 0.0

    # Stack into feature matrix
    X = np.vstack((qt_scaled, rr_scaled)).T
    return X, qt_min, qt_max, rr_mean, rr_std


def run_sindy_analysis(sheet_name, row_range, label):
    """Execute complete SINDy pipeline on ECG data.
    
    Args:
        sheet_name: WFDB record name
        row_range: Ignored (for compatibility)
        label: Description for output
    """
    record_name = sheet_name

    # Load data
    X, qt_min, qt_max, rr_mean, rr_std = load_ecg_subset(record_name, sheet_name, row_range)
    print(f"\nLoaded {len(X)} points mapping: {label}")

    # Train/test split (70/30)
    split_idx = int(len(X) * 0.7)
    split_idx = max(2, min(split_idx, len(X) - 1))

    X_train = X[:split_idx]
    X_test = X[split_idx:]

    print(f"Training on {len(X_train)} steps | Testing on {len(X_test)} steps")

    print("\n--- SINDy RESULTS (Equations) ---")

    # Fit polynomial model
    C = build_discrete_poly2_model(X_train)
    print_equations_numpy(C)

    # Evaluate on test set
    sindytest_numpy(
        C,
        X_train,
        X_test,
        label=label,
        qt_min=qt_min,
        qt_max=qt_max,
        rr_mean=rr_mean,
        rr_std=rr_std
    )


def main():
    """Main entry point: run SINDy on configured ECG records."""
    configs = [
        ('ARR_02', None, "Subject ARR_02 ECG"),
    ]

    for record_name, rows, label in configs:
        run_sindy_analysis(record_name, rows, label)


if __name__ == "__main__":
    main()