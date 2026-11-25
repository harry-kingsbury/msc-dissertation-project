"""ECG-QFM: Quantum Forecasting Model using Quantum Reservoir Computing

This module applies quantum reservoir computing to ECG data by:
1. Loading ECG records and extracting RR intervals and QT intervals
2. Creating quantum circuits that encode input signals as quantum states
3. Using quantum superposition to generate high-dimensional features
4. Training a ridge regression readout on these quantum features
5. Forecasting and evaluating accuracy

Quantum reservoir computing leverages quantum superposition for feature generation,
potentially capturing nonlinear patterns with fewer parameters.
"""

import os  # File operations
import numpy as np  # Numerical computing
import pandas as pd  # Data manipulation
from sklearn.preprocessing import StandardScaler  # Standardization
from sklearn.linear_model import RidgeCV  # Ridge regression with cross-validation
from sklearn.metrics import mean_squared_error  # MSE metric
from qiskit import QuantumCircuit  # Quantum circuit construction
from qiskit.quantum_info import Statevector  # Quantum state simulation
import wfdb  # WFDB toolkit
from wfdb import processing  # R-peak detection


class ECGQRC:
    """Quantum Forecasting Model for ECG data using quantum reservoir computing.
    
    Architecture:
    1. Quantum encoder: maps input states to quantum circuits
    2. Quantum reservoir: superposition generates high-dim features
    3. Classical readout: ridge regression from quantum features to targets
    """
    def __init__(self, n_qubits=3):
        """Initialize QFM with specified number of qubits.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit
        """
        self.n_qubits = n_qubits  # Number of quantum bits
        self.x_scaler = StandardScaler()  # Scaler for input features
        self.y_qt_scaler = StandardScaler()  # Scaler for QT outputs
        self.y_rr_scaler = StandardScaler()  # Scaler for RR outputs
        self.model_qt = None  # Ridge regression model for QT
        self.model_rr = None  # Ridge regression model for RR
        # Generate basis state labels (e.g., 000, 001, ..., 111 for 3 qubits)
        self.qf_names = [f"qf_{i:03b}" for i in range(2 ** n_qubits)]
        self._cache = {}  # Cache for quantum state computations

    def load_ecg_subset_wfdb(self, record_name):
        """Load ECG data from WFDB format and extract cardiac features.
        
        Args:
            record_name: Base name of WFDB record
        
        Returns:
            X: Feature matrix (n_samples, 2) - [QT_scaled, RR_scaled]
            qt_min, qt_max: QT bounds
            rr_mean, rr_std: RR normalization parameters
        """
        # Verify WFDB files exist
        if not os.path.exists(record_name + ".dat") or not os.path.exists(record_name + ".hea"):
            raise FileNotFoundError(f"WFDB files for record '{record_name}' not found.")

        # Load ECG signal
        rec = wfdb.rdrecord(record_name)
        ecg = rec.p_signal[:, 0]
        fs = rec.fs

        # Detect R-peaks
        r_peaks = processing.gqrs_detect(ecg, fs)

        # Calculate RR intervals
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / fs) * 1000.0
        rr_ms = rr_ms[np.isfinite(rr_ms)]

        # Outlier removal
        mask = (rr_ms > 300) & (rr_ms < 2000)
        rr_ms = rr_ms[mask]

        # Standardize RR intervals
        rr_mean = rr_ms.mean()
        rr_std = rr_ms.std()
        rr_scaled = (rr_ms - rr_mean) / (rr_std + 1e-8)

        # Placeholder QT interval (would need additional processing to extract)
        qt_scaled = np.zeros_like(rr_scaled)
        qt_min, qt_max = 0.0, 0.0

        # Stack into feature matrix
        X = np.vstack((qt_scaled, rr_scaled)).T
        return X, qt_min, qt_max, rr_mean, rr_std

    def minmax_pi(self, x):
        """Scale values to [0, π] range for quantum encoding.
        
        Quantum angles should be in [0, 2π). We use [0, π] for stability.
        
        Args:
            x: Input array
        
        Returns:
            x_scaled: Scaled to [0, π]
            xmin, xmax: Scaling bounds
        """
        x = np.asarray(x, dtype=float)
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        # If all values are the same, return zeros
        if np.isclose(xmax, xmin):
            return np.zeros_like(x), xmin, xmax
        # Linear scaling: (x - min) / (max - min) * π
        return (x - xmin) / (xmax - xmin) * np.pi, xmin, xmax

    def _normalize_state(self, state):
        """Normalize state values to valid quantum angle range [0, π].
        
        Handles NaN, inf values and clips to prevent quantum circuit errors.
        
        Args:
            state: Input state vector
        
        Returns:
            Normalized state in [0, π]
        """
        state = np.asarray(state, dtype=float)
        # Replace NaN/inf with safe values
        state = np.nan_to_num(state, nan=0.0, posinf=np.pi, neginf=0.0)
        # Clip to valid range
        return np.clip(state, 0.0, np.pi)

    def encode_input(self, state, memory):
        """Encode input and previous state into quantum superposition.
        
        Creates a quantum circuit that encodes the ECG signal as angles,
        applies rotations and entanglement, and extracts feature probabilities.
        
        Args:
            state: Current input state [QT, RR]
            memory: Previous state for context
        
        Returns:
            Probability distribution over computational basis states (2^n_qubits,)
        """
        # Normalize inputs
        state = self._normalize_state(state)
        memory = self._normalize_state(memory)
        
        # Create cache key to avoid recomputation
        key = (tuple(np.round(state, 6)), tuple(np.round(memory, 6)))
        if key in self._cache:
            return self._cache[key]

        # Build quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        # Initial encoding layer: rotate qubits based on input
        qc.ry(state[0], 0)  # QT on qubit 0
        qc.rz(state[1], 1)  # RR on qubit 1
        qc.ry(memory[0], 2)  # Memory on qubit 2
        # Entanglement layer: create correlations between qubits
        qc.cx(0, 1)  # CNOT: qubit 0 controls qubit 1
        qc.cx(1, 2)  # CNOT: qubit 1 controls qubit 2
        # Second encoding: more complex gates based on combined features
        qc.rx(0.5 * state[0] * state[1], 0)  # Interaction term
        qc.rz(memory[1], 2)  # Memory interaction
        qc.ry(0.25 * (state[0] + memory[0]), 1)  # Mixed rotation
        qc.cx(0, 2)  # Final entanglement

        # Extract quantum state and compute probabilities
        sv = Statevector.from_instruction(qc)  # Get statevector
        probs = np.abs(sv.data) ** 2  # Compute probabilities |amplitude|^2
        # Normalize (should be ~1 but enforce for numerical stability)
        probs = probs / (probs.sum() + 1e-12)
        # Cache result
        self._cache[key] = probs
        return probs

    def build_reservoir_states(self, X):
        """Build quantum reservoir feature matrix for all time steps.
        
        For each time step, encode the input and previous state into quantum features.
        This creates a high-dimensional representation of the dynamics.
        
        Args:
            X: Input trajectory (n_samples, 2)
        
        Returns:
            R: Quantum features (n_samples-1, 2^n_qubits)
        """
        states = []
        memory = np.zeros(2, dtype=float)  # Initialize memory
        # Encode each time step
        for t in range(len(X) - 1):
            states.append(self.encode_input(X[t], memory))
            memory = X[t].copy()  # Update memory for next step
        return np.asarray(states, dtype=float)

    def fit(self, X_train):
        """Train quantum forecasting model on training data.
        
        Process:
        1. Generate quantum features from input trajectory
        2. Standardize inputs and outputs
        3. Fit separate ridge regression models for QT and RR
        
        Args:
            X_train: Training trajectory (n_train, 2)
        """
        # Generate quantum features
        R_train = self.build_reservoir_states(X_train)
        # Target: next state (shifted by 1)
        y_train = X_train[1:1 + len(R_train)]

        # Standardize quantum features and targets
        R_train_s = self.x_scaler.fit_transform(R_train)
        y_qt_s = self.y_qt_scaler.fit_transform(y_train[:, 0].reshape(-1, 1)).ravel()
        y_rr_s = self.y_rr_scaler.fit_transform(y_train[:, 1].reshape(-1, 1)).ravel()

        # Train ridge regression models with cross-validation
        # Ridge uses L2 regularization to prevent overfitting
        alphas = np.logspace(-4, 2, 20)  # Range of regularization strengths
        self.model_qt = RidgeCV(alphas=alphas, fit_intercept=True)
        self.model_rr = RidgeCV(alphas=alphas, fit_intercept=True)
        self.model_qt.fit(R_train_s, y_qt_s)  # Fit QT model
        self.model_rr.fit(R_train_s, y_rr_s)  # Fit RR model

    def forecast(self, X_train, n_test):
        """Generate forecasts for n_test steps into future.
        
        Iteratively encodes current state into quantum features and predicts next state.
        
        Args:
            X_train: Training data (used for initialization)
            n_test: Number of steps to forecast
        
        Returns:
            preds: Forecasted trajectory (n_test, 2)
        """
        preds = []
        memory = X_train[-1].copy()  # Start with last training state as context
        current = X_train[-1].copy()  # Initialize prediction

        for _ in range(n_test):
            # Encode current state with memory context
            feat = self.encode_input(current, memory)
            # Standardize features using training scalers
            feat_s = self.x_scaler.transform(feat.reshape(1, -1))

            # Predict next QT and RR using trained models
            qt_s = self.model_qt.predict(feat_s)[0]
            rr_s = self.model_rr.predict(feat_s)[0]

            # Inverse transform to get values in original scale
            qt = self.y_qt_scaler.inverse_transform([[qt_s]])[0, 0]
            rr = self.y_rr_scaler.inverse_transform([[rr_s]])[0, 0]

            # Clip to valid range to prevent numerical instability
            qt = float(np.clip(qt, 0.0, np.pi))
            rr = float(np.clip(rr, 0.0, np.pi))

            preds.append([qt, rr])
            memory = current.copy()  # Update memory
            current = np.array([qt, rr], dtype=float)  # Update current state

        return np.asarray(preds, dtype=float)

    def evaluate(self, pred_scaled, true_scaled, label="Heart Condition",
                 rr_mean=None, rr_std=None):
        """Evaluate model performance on test data.
        
        Computes MSE and assesses forecast stability.
        
        Args:
            pred_scaled: Predictions
            true_scaled: Ground truth
            label: Description
            rr_mean, rr_std: Scaling parameters for physical units
        """
        # Compute MSE for each dimension
        qt_mse_scaled = mean_squared_error(true_scaled[:, 0], pred_scaled[:, 0])
        rr_mse_scaled = mean_squared_error(true_scaled[:, 1], pred_scaled[:, 1])

        print(f"\n{'='*50}\nVALIDATING QFM MODEL: {label}\n{'='*50}")
        print("QFM Forecasting Accuracy Metrics:")
        print(f"  |-- QT Scaled MSE: {qt_mse_scaled:.8f}")
        print(f"  |-- RR Scaled MSE: {rr_mse_scaled:.8f}")

        # Convert to physical units if scaling provided
        if rr_mean is not None and rr_std is not None:
            pred_rr = pred_scaled[:, 1] * rr_std + rr_mean
            true_rr = true_scaled[:, 1] * rr_std + rr_mean
            rr_error = mean_squared_error(true_rr, pred_rr)
            print(f"  |-- RR Interval Forecast Error: {rr_error:.8f} ms^2")

        # Assess stability (lower MSE = better)
        if rr_mse_scaled < 0.5:
            print("Stable System: The QFM model successfully forecasted future cardiac dynamics.")
        else:
            print("High Deviation: The QFM forecast drifted from the true test segment.")
        print("=" * 50 + "\n")


def run_pipeline(record_name, label):
    """Execute complete QFM pipeline on ECG data.
    
    Args:
        record_name: WFDB record base name
        label: Description for output messages
    """
    # Initialize quantum forecasting model
    engine = ECGQRC(n_qubits=3)

    # Load ECG data
    X, qt_min, qt_max, rr_mean, rr_std = engine.load_ecg_subset_wfdb(record_name)

    # Scale to [0, π] range for quantum encoding
    qt_scaled_pi, _, _ = engine.minmax_pi(X[:, 0])
    rr_scaled_pi, _, _ = engine.minmax_pi(X[:, 1])
    X_pi = np.vstack([qt_scaled_pi, rr_scaled_pi]).T

    # Train/test split (70/30)
    split_idx = int(len(X_pi) * 0.7)
    split_idx = max(5, min(split_idx, len(X_pi) - 1))
    X_train, X_test = X_pi[:split_idx], X_pi[split_idx:]

    print(f"\nProcessing {label}...")
    print(f"Training on {len(X_train)} steps | Testing on {len(X_test)} steps")
    print("Step 1: Building quantum reservoir states...")
    # Train quantum model
    engine.fit(X_train)
    print("Step 2: Training ridge readout...")

    # Display learned readout equations
    print("\n" + "=" * 60 + f"\nDISCOVERED QFM READOUT ({label.upper()}):\n" + "=" * 60)
    # Format QT equations
    qt_terms = [f"({c:+.4f} * {n})" for c, n in zip(engine.model_qt.coef_, engine.qf_names) if abs(c) > 1e-4]
    # Format RR equations
    rr_terms = [f"({c:+.4f} * {n})" for c, n in zip(engine.model_rr.coef_, engine.qf_names) if abs(c) > 1e-4]
    print("QT_scaled_z[k+1] = " + " ".join(qt_terms) + f" + ({engine.model_qt.intercept_:.4f})")
    print("RR_scaled_z[k+1] = " + " ".join(rr_terms) + f" + ({engine.model_rr.intercept_:.4f})")
    print("=" * 60)

    # Forecast on test set
    pred_scaled = engine.forecast(X_train, len(X_test))
    # Evaluate performance
    engine.evaluate(pred_scaled, X_test, label=label, rr_mean=rr_mean, rr_std=rr_std)


def main():
    """Main entry point: run QFM on configured ECG records."""
    print("=== STARTING QUANTUM MASTER EXPERIMENT (WFDB ECG) ===")
    configs = [
        ('ARR_02', "Subject ARR_02 ECG (QFM)"),
    ]

    for record_name, label in configs:
        run_pipeline(record_name, label)


if __name__ == "__main__":
    main()