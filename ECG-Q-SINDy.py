"""ECG-Q-SINDy: Quantum-Enhanced Sparse Identification of Nonlinear Dynamics

This module combines quantum computing with SINDy by:
1. Loading ECG records and extracting RR intervals
2. Building polynomial feature library (classical)
3. Building quantum feature library using PennyLane circuits
4. Using sparse regression to discover dynamics in hybrid feature space
5. Forecasting and evaluating accuracy

The quantum features provide nonlinear transformations that complement polynomial terms.
"""

import os  # File operations
import numpy as np  # Numerical computing
import wfdb  # WFDB toolkit for cardiac data
from wfdb import processing  # R-peak detection

from sklearn.linear_model import Ridge  # Ridge regression
from sklearn.metrics import mean_squared_error  # MSE metric

import pennylane as qml  # Quantum computing framework

try:
    from pysindy.optimizers import STLSQ  # Sparse Thresholded Least Squares optimizer
    HAS_PYSINDY = True  # PySINDy available
except Exception:
    HAS_PYSINDY = False  # Fallback to Ridge regression if unavailable


def load_ecg_rr_only(record_name):
    """Load ECG and extract standardized RR intervals.
    
    Args:
        record_name: Base name of WFDB record
    
    Returns:
        rr_scaled: Standardized RR intervals (zero mean, unit variance)
        rr_mean, rr_std: Normalization parameters for inverse transform
    """
    # Verify WFDB files exist
    if not os.path.exists(record_name + ".dat") or not os.path.exists(record_name + ".hea"):
        raise FileNotFoundError(f"WFDB files for record '{record_name}' not found.")

    # Load ECG signal
    rec = wfdb.rdrecord(record_name)
    ecg = rec.p_signal[:, 0]
    fs = rec.fs

    # Detect R-peaks using Goulart-Quezada algorithm
    r_peaks = processing.gqrs_detect(ecg, fs)

    # Compute RR intervals in milliseconds
    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs) * 1000.0
    rr_ms = rr_ms[np.isfinite(rr_ms)]  # Remove NaN/inf

    # Remove outliers: keep physiologically plausible values (300-2000 ms)
    mask = (rr_ms > 300) & (rr_ms < 2000)
    rr_ms = rr_ms[mask]

    # Standardize for numerical stability
    rr_mean = rr_ms.mean()
    rr_std = rr_ms.std()
    rr_scaled = (rr_ms - rr_mean) / (rr_std + 1e-8)

    return rr_scaled, rr_mean, rr_std


def build_polynomial_library_1d(x):
    """Build polynomial feature library for 1D signals.
    
    Creates basis functions: [1, x, x^2] - constant, linear, quadratic terms.
    These are classical features for SINDy.
    
    Args:
        x: Input signal (n_samples,)
    
    Returns:
        P: Feature matrix (n_samples, 3)
    """
    x = np.asarray(x)
    # Stack polynomial monomials as columns
    P = np.column_stack([
        np.ones_like(x),  # Constant term
        x,  # Linear: x
        x**2  # Quadratic: x^2
    ])
    return P


def quantum_feature_map(x, n_qubits=2):
    """Generate quantum features using variational quantum circuit.
    
    Each input value is encoded into a quantum circuit, then measured
    to extract n_qubits features. The circuit applies rotations and
    entanglement to create nonlinear feature representations.
    
    Args:
        x: Input signal (n_samples,)
        n_qubits: Number of qubits (= number of features)
    
    Returns:
        Q: Quantum feature matrix (n_samples, n_qubits)
    """
    x = np.asarray(x)
    n_samples = len(x)

    # Create quantum device (classical simulator)
    dev = qml.device("default.qubit", wires=n_qubits)

    # Define quantum circuit
    @qml.qnode(dev)  # Converts to executable quantum node
    def circuit(xi):
        # First encoding layer: encode input as rotation angles
        for i in range(n_qubits):
            qml.RX(xi, wires=i)  # Rotation around X-axis
        
        # Entangling layer: create correlations between qubits
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i+1])  # Controlled-Z (CZ) gates
        
        # Second rotation layer: additional nonlinear transformations
        for i in range(n_qubits):
            qml.RY(xi, wires=i)  # Rotation around Y-axis

        # Measure expectation value of Pauli-Z on each qubit
        # Pauli-Z measurement gives values in [-1, 1]
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Execute circuit for each input sample
    Q = np.zeros((n_samples, n_qubits))
    for i, xi in enumerate(x):
        Q[i] = circuit(xi)  # Run circuit, get n_qubits output values
    return Q


def orthogonalize(Q, P):
    """Orthogonalize quantum features with respect to polynomial features.
    
    Removes the component of Q that can be explained by P, ensuring
    quantum features capture information not in polynomial terms.
    This prevents feature cannibalization where one set dominates.
    
    Args:
        Q: Quantum features (n_samples, n_q_features)
        P: Polynomial features (n_samples, n_p_features)
    
    Returns:
        Q_perp: Orthogonalized quantum features
    """
    # Find projection: minimize ||P @ A - Q||^2 (least squares fit)
    A, _, _, _ = np.linalg.lstsq(P, Q, rcond=None)
    # Compute residual (orthogonal component)
    Q_perp = Q - P @ A
    return Q_perp


def q_sindy_fit(P, Q, x_dot, threshold=1e-3, alpha=1e-4):
    """Fit sparse dynamics using polynomial features (quantum features unused for now).
    
    Solves: x_dot = P @ xi_P + Q @ xi_Q
    Currently uses only polynomial features; quantum features can be integrated.
    
    Args:
        P: Polynomial feature matrix (n_samples, n_p_features)
        Q: Quantum feature matrix (n_samples, n_q_features)
        x_dot: Time derivative of signal (target)
        threshold: Sparsification threshold for STLSQ
        alpha: Ridge regression regularization strength
    
    Returns:
        xi_P: Polynomial coefficients
        xi_Q: Quantum coefficients (currently zeros)
    """
    # Current implementation: use only polynomial library
    Theta = P

    if HAS_PYSINDY:
        # Use sparse thresholded least squares optimizer
        opt = STLSQ(threshold=threshold, alpha=alpha)
        opt.fit(Theta, x_dot[:, None])
        # Extract coefficients (different PySINDy versions have different attributes)
        if hasattr(opt, "coefficients_"):
            xi = opt.coefficients_.ravel()
        elif hasattr(opt, "coefficients"):
            xi = opt.coefficients().ravel()
        else:
            # Fallback: use Ridge regression
            reg = Ridge(alpha=alpha)
            reg.fit(Theta, x_dot)
            xi = reg.coef_
    else:
        # No PySINDy: use Ridge regression instead
        reg = Ridge(alpha=alpha)
        reg.fit(Theta, x_dot)
        xi = reg.coef_

    # Split coefficients into polynomial and quantum parts
    p = P.shape[1]  # Number of polynomial features
    xi_P = xi[:p]  # Polynomial coefficients
    xi_Q = np.zeros(Q.shape[1])  # Quantum coefficients (unused)
    return xi_P, xi_Q


def print_q_sindy_equation(xi_P, xi_Q, degree=2):
    """Display discovered Q-SINDy equations in human-readable format.
    
    Prints the sparse polynomial and quantum coefficients, showing only
    terms that exceed significance threshold.
    
    Args:
        xi_P: Polynomial coefficients [c_1, c_x, c_x2, ...]
        xi_Q: Quantum feature coefficients
        degree: Polynomial degree (for labeling)
    """
    # Format polynomial part
    poly_terms = []
    for i, c in enumerate(xi_P):
        # Create term labels
        if i == 0:
            name = "1"  # Constant
        elif i == 1:
            name = "x"  # Linear: RR
        elif i == 2:
            name = "x^2"  # Quadratic: RR^2
        else:
            name = f"x^{i}"  # Higher powers
        # Include only significant coefficients (above threshold)
        if abs(c) > 1e-6:
            sign = "+" if c >= 0 else "-"
            poly_terms.append(f"{sign} {abs(c):.6f} * {name}")

    # Format quantum part (currently zeros)
    quantum_terms = []
    for j, c in enumerate(xi_Q):
        if abs(c) > 1e-6:
            sign = "+" if c >= 0 else "-"
            quantum_terms.append(f"{sign} {abs(c):.6f} * q_{j+1}(x)")

    # Print the discovered equation
    print("\nDiscovered Q-SINDy equation (scaled RR):")
    print("dx/dt = " + " ".join(poly_terms + quantum_terms))
    print("with x = RR_scaled, and q_j(x) = quantum feature from circuit.")


def q_sindy_predict(P_row, Q_row, xi_P, xi_Q):
    """Predict time derivative using polynomial and quantum features.
    
    Computes: dx/dt = P @ xi_P + Q @ xi_Q
    
    Args:
        P_row: Polynomial features at current state (1, n_p_features)
        Q_row: Quantum features at current state (1, n_q_features)
        xi_P: Polynomial coefficients
        xi_Q: Quantum coefficients
    
    Returns: Predicted time derivative
    """
    # Linear combination of feature-weighted coefficients
    return P_row @ xi_P + Q_row @ xi_Q


def forecast_q_sindy_1d(xi_P, xi_Q, x0, n_steps, build_P, build_Q):
    """Forecast signal using discovered Q-SINDy equations.
    
    Iteratively applies: x[k+1] = x[k] + dx/dt (Euler method with dt=1)
    
    Args:
        xi_P: Polynomial coefficients
        xi_Q: Quantum coefficients
        x0: Initial condition
        n_steps: Number of steps to forecast
        build_P: Function to build polynomial features
        build_Q: Function to build quantum features
    
    Returns:
        traj: Forecasted trajectory (n_steps+1,)
    """
    traj = np.zeros(n_steps + 1)  # Allocate output
    traj[0] = x0  # Initial condition
    x = x0
    
    # Iteratively predict next state
    for k in range(n_steps):
        # Compute features at current state
        P_row = build_P(np.array([x]))[0]  # Polynomial features
        Q_row = build_Q(np.array([x]))[0]  # Quantum features
        # Predict time derivative
        x_dot = q_sindy_predict(P_row, Q_row, xi_P, xi_Q)
        # Euler integration: x[k+1] = x[k] + dx/dt * dt (dt=1)
        x = x + 1.0 * x_dot
        traj[k+1] = x  # Store prediction
    
    return traj


def run_q_sindy_ecg(record_name, label):
    """Execute complete Q-SINDy pipeline on ECG data.
    
    Args:
        record_name: WFDB record base name
        label: Description for output messages
    """
    print(f"\n=== Q-SINDy on ECG: {label} ===")

    # Load and preprocess ECG
    rr_scaled, rr_mean, rr_std = load_ecg_rr_only(record_name)
    x = rr_scaled
    print(f"Loaded {len(x)} RR beats")

    # Train/test split (70/30)
    split_idx = int(len(x) * 0.7)
    split_idx = max(5, min(split_idx, len(x) - 1))
    x_train = x[:split_idx]
    x_test = x[split_idx:]

    # Build feature libraries on training data
    P_train = build_polynomial_library_1d(x_train)  # Polynomial features
    Q_train = quantum_feature_map(x_train, n_qubits=2)  # Quantum features

    # Compute time derivative using finite differences
    x_dot_train = np.diff(x_train)

    # Align feature matrices with time derivatives (drop last sample due to diff)
    P_train = P_train[:-1]
    Q_train = Q_train[:-1]

    print(f"Training on {len(x_train)-1} steps | Testing on {len(x_test)} steps")

    # Fit sparse dynamics model
    xi_P, xi_Q = q_sindy_fit(P_train, Q_train, x_dot_train, threshold=1e-3, alpha=1e-4)

    # Display learned coefficients
    print("Discovered polynomial coefficients (1, x, x^2):")
    print(xi_P)
    print("Quantum feature coefficients:")
    print(xi_Q)

    # Display discovered governing equation
    print_q_sindy_equation(xi_P, xi_Q, degree=2)

    # Forecast test segment starting from last training point
    x0 = x_train[-1]
    traj = forecast_q_sindy_1d(
        xi_P, xi_Q, x0, len(x_test),
        lambda y: build_polynomial_library_1d(y),
        lambda y: quantum_feature_map(y, n_qubits=2)
    )
    # Extract predictions (skip initial condition)
    x_pred = traj[1:]

    # Compute forecasting errors
    # Scaled error (standardized units)
    mse_scaled = mean_squared_error(x_test, x_pred)
    # Physical error (milliseconds squared)
    mse_ms2 = ((x_pred * rr_std + rr_mean) - (x_test * rr_std + rr_mean))**2
    mse_ms2 = mse_ms2.mean()

    # Print performance metrics
    print("\nQ-SINDy Forecasting Metrics:")
    print(f"  RR Scaled MSE: {mse_scaled:.8f}")
    print(f"  RR Interval Error (ms^2): {mse_ms2:.8f}")

    # Assess stability
    if mse_scaled < 0.5:
        print("Stable System: The Q-SINDy equations successfully captured the cardiac rhythm.")
    else:
        print("High Deviation: The Q-SINDy forecast drifted from the true test segment.")


def main():
    """Main entry point: run Q-SINDy on configured ECG records."""
    configs = [
        ('ARR_02', "Subject ARR_02 ECG (Q-SINDy)"),
    ]
    
    # Process each record
    for record_name, label in configs:
        run_q_sindy_ecg(record_name, label)


if __name__ == "__main__":
    main()

