"""
Quantum Computing preparation for portfolio optimization.

Implements quantum-inspired algorithms and QAOA (Quantum Approximate Optimization Algorithm)
for portfolio optimization problems, preparing for quantum hardware integration.
"""
from __future__ import annotations

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from scipy.optimize import minimize
from scipy.linalg import expm
import itertools

logger = logging.getLogger(__name__)


@dataclass
class QuantumCircuit:
    """Quantum circuit representation for optimization."""
    n_qubits: int
    depth: int
    gates: List[Tuple[str, List[int], float]]  # (gate_type, qubits, parameter)
    measurements: Dict[str, int] = field(default_factory=dict)
    statevector: Optional[np.ndarray] = None


@dataclass
class QAOAParameters:
    """QAOA algorithm parameters."""
    p: int = 3  # Number of QAOA layers
    beta: List[float] = field(default_factory=list)  # Mixing parameters
    gamma: List[float] = field(default_factory=list)  # Problem parameters
    optimization_method: str = "COBYLA"
    max_iterations: int = 1000
    tolerance: float = 1e-6


@dataclass
class QuantumOptimizationResult:
    """Result from quantum optimization."""
    optimal_solution: np.ndarray
    optimal_value: float
    quantum_circuit: QuantumCircuit
    convergence_history: List[float]
    execution_time_ms: float
    algorithm: str
    parameters: Dict[str, Any]


class QuantumStatevectorSimulator:
    """Simulate quantum circuits using statevector representation."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.statevector = self._initialize_statevector()

    def _initialize_statevector(self) -> np.ndarray:
        """Initialize quantum state to |00...0>."""
        statevector = np.zeros(2 ** self.n_qubits, dtype=complex)
        statevector[0] = 1.0
        return statevector

    def apply_hadamard(self, qubit: int):
        """Apply Hadamard gate to a qubit."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_qubit_gate(H, qubit)

    def apply_rx(self, qubit: int, theta: float):
        """Apply rotation around X-axis."""
        Rx = np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ])
        self._apply_single_qubit_gate(Rx, qubit)

    def apply_ry(self, qubit: int, theta: float):
        """Apply rotation around Y-axis."""
        Ry = np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
        self._apply_single_qubit_gate(Ry, qubit)

    def apply_rz(self, qubit: int, theta: float):
        """Apply rotation around Z-axis."""
        Rz = np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ])
        self._apply_single_qubit_gate(Rz, qubit)

    def apply_cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        for i in range(2 ** self.n_qubits):
            if self._get_bit(i, control) == 1:
                j = i ^ (1 << target)  # Flip target bit
                if i < j:
                    self.statevector[i], self.statevector[j] = (
                        self.statevector[j], self.statevector[i]
                    )

    def apply_rzz(self, qubit1: int, qubit2: int, theta: float):
        """Apply ZZ interaction gate."""
        for i in range(2 ** self.n_qubits):
            parity = (self._get_bit(i, qubit1) + self._get_bit(i, qubit2)) % 2
            phase = np.exp(-1j * theta / 2) if parity == 0 else np.exp(1j * theta / 2)
            self.statevector[i] *= phase

    def _apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """Apply single-qubit gate to the statevector."""
        new_statevector = np.zeros_like(self.statevector)

        for i in range(2 ** self.n_qubits):
            bit_value = self._get_bit(i, qubit)
            i_flipped = i ^ (1 << qubit)

            if bit_value == 0:
                new_statevector[i] += gate[0, 0] * self.statevector[i]
                new_statevector[i_flipped] += gate[1, 0] * self.statevector[i]
            else:
                new_statevector[i_flipped] += gate[0, 1] * self.statevector[i]
                new_statevector[i] += gate[1, 1] * self.statevector[i]

        self.statevector = new_statevector

    def _get_bit(self, integer: int, bit_position: int) -> int:
        """Get bit value at position."""
        return (integer >> bit_position) & 1

    def measure_all(self, n_shots: int = 1000) -> Dict[str, int]:
        """Measure all qubits and return counts."""
        probabilities = np.abs(self.statevector) ** 2
        outcomes = np.random.choice(
            2 ** self.n_qubits,
            size=n_shots,
            p=probabilities
        )

        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def get_expectation_value(self, observable: np.ndarray) -> float:
        """Calculate expectation value of an observable."""
        return np.real(np.dot(np.conj(self.statevector), observable @ self.statevector))


class QAOAOptimizer:
    """Quantum Approximate Optimization Algorithm for combinatorial optimization."""

    def __init__(self, parameters: QAOAParameters):
        self.parameters = parameters

    def optimize_portfolio(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 1.0,
        n_assets: Optional[int] = None
    ) -> QuantumOptimizationResult:
        """Optimize portfolio using QAOA."""
        if n_assets is None:
            n_assets = min(len(returns), 10)  # Limit to 10 for simulation

        logger.info(f"Starting QAOA portfolio optimization for {n_assets} assets")

        # Formulate as QUBO problem
        Q = self._formulate_qubo(returns[:n_assets], covariance[:n_assets, :n_assets], risk_aversion)

        # Initialize parameters
        if not self.parameters.beta:
            self.parameters.beta = np.random.uniform(0, np.pi, self.parameters.p).tolist()
        if not self.parameters.gamma:
            self.parameters.gamma = np.random.uniform(0, 2*np.pi, self.parameters.p).tolist()

        # Run QAOA
        start_time = datetime.now()
        circuit, optimal_params, history = self._run_qaoa(Q, n_assets)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        # Extract solution
        measurements = circuit.measurements
        best_solution = self._extract_best_solution(measurements)

        # Calculate optimal value
        optimal_value = self._calculate_portfolio_value(
            best_solution, returns[:n_assets], covariance[:n_assets, :n_assets], risk_aversion
        )

        return QuantumOptimizationResult(
            optimal_solution=best_solution,
            optimal_value=optimal_value,
            quantum_circuit=circuit,
            convergence_history=history,
            execution_time_ms=execution_time,
            algorithm="QAOA",
            parameters={
                "p": self.parameters.p,
                "risk_aversion": risk_aversion,
                "n_assets": n_assets
            }
        )

    def _formulate_qubo(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float
    ) -> np.ndarray:
        """Formulate portfolio optimization as QUBO problem."""
        n = len(returns)

        # QUBO matrix: maximize returns - risk_aversion * variance
        Q = np.zeros((n, n))

        # Linear terms (returns)
        for i in range(n):
            Q[i, i] = -returns[i]

        # Quadratic terms (risk)
        for i in range(n):
            for j in range(n):
                Q[i, j] += risk_aversion * covariance[i, j]

        return Q

    def _run_qaoa(
        self,
        Q: np.ndarray,
        n_qubits: int
    ) -> Tuple[QuantumCircuit, np.ndarray, List[float]]:
        """Run QAOA algorithm."""
        history = []

        def qaoa_objective(params):
            """Objective function for QAOA."""
            beta = params[:self.parameters.p]
            gamma = params[self.parameters.p:]

            # Create and execute circuit
            circuit = self._create_qaoa_circuit(n_qubits, beta, gamma, Q)

            # Calculate expectation value
            expectation = self._calculate_expectation(circuit, Q)
            history.append(-expectation)  # Minimize negative expectation

            return -expectation

        # Initial parameters
        initial_params = np.concatenate([self.parameters.beta, self.parameters.gamma])

        # Optimize parameters
        result = minimize(
            qaoa_objective,
            initial_params,
            method=self.parameters.optimization_method,
            options={
                'maxiter': self.parameters.max_iterations,
                'tol': self.parameters.tolerance
            }
        )

        # Create final circuit with optimal parameters
        optimal_beta = result.x[:self.parameters.p]
        optimal_gamma = result.x[self.parameters.p:]
        final_circuit = self._create_qaoa_circuit(n_qubits, optimal_beta, optimal_gamma, Q)

        return final_circuit, result.x, history

    def _create_qaoa_circuit(
        self,
        n_qubits: int,
        beta: np.ndarray,
        gamma: np.ndarray,
        Q: np.ndarray
    ) -> QuantumCircuit:
        """Create QAOA quantum circuit."""
        circuit = QuantumCircuit(n_qubits=n_qubits, depth=2*self.parameters.p+1, gates=[])

        # Initialize simulator
        simulator = QuantumStatevectorSimulator(n_qubits)

        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            simulator.apply_hadamard(i)
            circuit.gates.append(("H", [i], 0))

        # Apply QAOA layers
        for layer in range(self.parameters.p):
            # Problem Hamiltonian
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    if abs(Q[i, j]) > 1e-10:
                        simulator.apply_rzz(i, j, 2 * gamma[layer] * Q[i, j])
                        circuit.gates.append(("RZZ", [i, j], 2 * gamma[layer] * Q[i, j]))

            # Mixing Hamiltonian
            for i in range(n_qubits):
                simulator.apply_rx(i, 2 * beta[layer])
                circuit.gates.append(("RX", [i], 2 * beta[layer]))

        # Measure
        circuit.measurements = simulator.measure_all(n_shots=1000)
        circuit.statevector = simulator.statevector

        return circuit

    def _calculate_expectation(self, circuit: QuantumCircuit, Q: np.ndarray) -> float:
        """Calculate expectation value for QUBO problem."""
        expectation = 0.0
        total_counts = sum(circuit.measurements.values())

        for bitstring, count in circuit.measurements.items():
            # Convert bitstring to solution vector
            solution = np.array([int(b) for b in bitstring])

            # Calculate QUBO value
            value = solution @ Q @ solution

            # Add weighted contribution
            expectation += value * (count / total_counts)

        return expectation

    def _extract_best_solution(self, measurements: Dict[str, int]) -> np.ndarray:
        """Extract best solution from measurements."""
        best_bitstring = max(measurements, key=measurements.get)
        return np.array([int(b) for b in best_bitstring])

    def _calculate_portfolio_value(
        self,
        solution: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float
    ) -> float:
        """Calculate portfolio value for given solution."""
        portfolio_return = np.dot(solution, returns)
        portfolio_risk = solution @ covariance @ solution
        return portfolio_return - risk_aversion * portfolio_risk


class VQEOptimizer:
    """Variational Quantum Eigensolver for optimization problems."""

    def __init__(self, n_qubits: int, ansatz_depth: int = 3):
        self.n_qubits = n_qubits
        self.ansatz_depth = ansatz_depth

    def find_ground_state(
        self,
        hamiltonian: np.ndarray,
        initial_params: Optional[np.ndarray] = None
    ) -> QuantumOptimizationResult:
        """Find ground state of Hamiltonian using VQE."""
        logger.info(f"Starting VQE for {self.n_qubits} qubits")

        start_time = datetime.now()

        # Initialize parameters
        n_params = self.n_qubits * self.ansatz_depth * 3  # 3 rotation angles per qubit per layer
        if initial_params is None:
            initial_params = np.random.uniform(0, 2*np.pi, n_params)

        history = []

        def vqe_objective(params):
            """VQE objective function."""
            circuit = self._create_ansatz_circuit(params)
            expectation = self._calculate_hamiltonian_expectation(circuit, hamiltonian)
            history.append(expectation)
            return expectation

        # Optimize
        result = minimize(
            vqe_objective,
            initial_params,
            method='COBYLA',
            options={'maxiter': 500}
        )

        # Create final circuit
        final_circuit = self._create_ansatz_circuit(result.x)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        # Extract eigenstate
        eigenstate = final_circuit.statevector
        eigenvalue = result.fun

        return QuantumOptimizationResult(
            optimal_solution=np.abs(eigenstate) ** 2,
            optimal_value=eigenvalue,
            quantum_circuit=final_circuit,
            convergence_history=history,
            execution_time_ms=execution_time,
            algorithm="VQE",
            parameters={"ansatz_depth": self.ansatz_depth}
        )

    def _create_ansatz_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """Create parameterized ansatz circuit."""
        circuit = QuantumCircuit(n_qubits=self.n_qubits, depth=self.ansatz_depth, gates=[])
        simulator = QuantumStatevectorSimulator(self.n_qubits)

        param_idx = 0

        for layer in range(self.ansatz_depth):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                simulator.apply_ry(qubit, params[param_idx])
                circuit.gates.append(("RY", [qubit], params[param_idx]))
                param_idx += 1

                simulator.apply_rz(qubit, params[param_idx])
                circuit.gates.append(("RZ", [qubit], params[param_idx]))
                param_idx += 1

            # Entangling gates
            for qubit in range(self.n_qubits - 1):
                simulator.apply_cnot(qubit, qubit + 1)
                circuit.gates.append(("CNOT", [qubit, qubit + 1], 0))

            # Additional rotations
            for qubit in range(self.n_qubits):
                simulator.apply_ry(qubit, params[param_idx])
                circuit.gates.append(("RY", [qubit], params[param_idx]))
                param_idx += 1

        circuit.statevector = simulator.statevector
        return circuit

    def _calculate_hamiltonian_expectation(
        self,
        circuit: QuantumCircuit,
        hamiltonian: np.ndarray
    ) -> float:
        """Calculate expectation value of Hamiltonian."""
        statevector = circuit.statevector
        return np.real(np.dot(np.conj(statevector), hamiltonian @ statevector))


class QuantumPortfolioManager:
    """Manager for quantum-enhanced portfolio optimization."""

    def __init__(self):
        self.qaoa_optimizer = None
        self.vqe_optimizer = None
        self.optimization_history: List[QuantumOptimizationResult] = []

    def optimize_with_qaoa(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 1.0,
        p: int = 3
    ) -> QuantumOptimizationResult:
        """Optimize portfolio using QAOA."""
        params = QAOAParameters(p=p)
        self.qaoa_optimizer = QAOAOptimizer(params)

        result = self.qaoa_optimizer.optimize_portfolio(
            returns, covariance, risk_aversion
        )

        self.optimization_history.append(result)
        return result

    def optimize_with_vqe(
        self,
        cost_matrix: np.ndarray,
        n_qubits: int,
        ansatz_depth: int = 3
    ) -> QuantumOptimizationResult:
        """Optimize using VQE."""
        self.vqe_optimizer = VQEOptimizer(n_qubits, ansatz_depth)

        # Convert cost matrix to Hamiltonian
        hamiltonian = self._cost_to_hamiltonian(cost_matrix, n_qubits)

        result = self.vqe_optimizer.find_ground_state(hamiltonian)

        self.optimization_history.append(result)
        return result

    def _cost_to_hamiltonian(
        self,
        cost_matrix: np.ndarray,
        n_qubits: int
    ) -> np.ndarray:
        """Convert cost matrix to quantum Hamiltonian."""
        dim = 2 ** n_qubits
        hamiltonian = np.zeros((dim, dim), dtype=complex)

        # Map cost matrix to diagonal Hamiltonian
        for i in range(min(dim, len(cost_matrix))):
            if i < len(cost_matrix):
                hamiltonian[i, i] = np.sum(cost_matrix[i])

        return hamiltonian

    def benchmark_quantum_vs_classical(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float = 1.0
    ) -> Dict[str, Any]:
        """Benchmark quantum vs classical optimization."""
        n_assets = min(len(returns), 8)  # Limit for simulation

        # Quantum optimization
        quantum_start = datetime.now()
        quantum_result = self.optimize_with_qaoa(
            returns[:n_assets],
            covariance[:n_assets, :n_assets],
            risk_aversion
        )
        quantum_time = (datetime.now() - quantum_start).total_seconds()

        # Classical optimization (brute force for small problems)
        classical_start = datetime.now()
        classical_solution, classical_value = self._classical_optimize(
            returns[:n_assets],
            covariance[:n_assets, :n_assets],
            risk_aversion
        )
        classical_time = (datetime.now() - classical_start).total_seconds()

        return {
            "quantum": {
                "solution": quantum_result.optimal_solution.tolist(),
                "value": quantum_result.optimal_value,
                "time_seconds": quantum_time,
                "algorithm": "QAOA"
            },
            "classical": {
                "solution": classical_solution.tolist(),
                "value": classical_value,
                "time_seconds": classical_time,
                "algorithm": "brute_force"
            },
            "speedup": classical_time / quantum_time if quantum_time > 0 else 0,
            "n_assets": n_assets
        }

    def _classical_optimize(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_aversion: float
    ) -> Tuple[np.ndarray, float]:
        """Classical brute-force optimization for comparison."""
        n = len(returns)
        best_solution = None
        best_value = float('-inf')

        # Try all possible binary combinations
        for combination in itertools.product([0, 1], repeat=n):
            solution = np.array(combination)
            if np.sum(solution) > 0:  # Avoid empty portfolio
                value = np.dot(solution, returns) - risk_aversion * (solution @ covariance @ solution)
                if value > best_value:
                    best_value = value
                    best_solution = solution

        return best_solution, best_value


__all__ = [
    "QuantumPortfolioManager",
    "QAOAOptimizer",
    "VQEOptimizer",
    "QuantumCircuit",
    "QAOAParameters",
    "QuantumOptimizationResult",
    "QuantumStatevectorSimulator"
]