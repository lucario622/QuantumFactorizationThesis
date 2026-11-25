# coding:utf-8
"""
Using QAOA to get better solution than Babai's algorithm in Schnorr's algorithm.

Qiskit (>= 1.0 / 2.x) port of the original MindQuantum implementation.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


def dot(v1, v2) -> float:
    """Calculate the inner product of two vector.
    """
    return np.ravel(v1) @ np.ravel(v2)


def norm2(v):
    """Calculate the square of norm for input vector.
    """
    return dot(v, v)


@dataclass
class QAOAConfig:
    """Configuration for `QAOASearch`"""
    n_layer: int = 4                # Layers of QAOA circuit.
    # Optimizer for QAOA, only support Adam now.
    optimizer: str = "adam"
    learning_rate: float = 0.002    # Learning rate of optimizer.
    max_iter: int = 1000            # Number of iterations in QAOA.
    n_sample_shot: int = 1000       # Number of samples.
    verbose: int = 0                # Output log if `verbose` greater than 0.
    seed: Optional[int] = None      # Seed for initial weights and sampling.


class QAOASearch:
    """QAOA Search in SQIF algorithm.
    """

    def __init__(self, config: QAOAConfig):
        """
        Args:
            config: The QAOA configuration.
        """
        self.config = config
        self.verbose = config.verbose
        self.rng = np.random.default_rng(config.seed)

    def _get_hamil_coef(self, mat_d, diff, s):
        """Get coefficients for hamiltonian.
        Args:
            mat_d (np.ndarray): shape=(m,n). LLL-deduced basis.
            diff (np.ndarray): shape=(m,1). Target minus closest vector, column vector.
            s (np.ndarray): shape=(n). s[i] means symbol of ith variable. 1 means cj <= muj.
                -1 means cj > muj. See formula (S48) in paper.
        """
        n = mat_d.shape[1]
        mat_f = np.zeros((n, n))      # Coefficient matrix.
        g = np.zeros(n)
        l = 0.0

        mat_d = mat_d.copy()
        diff = diff.copy()

        mat_d *= s
        diff += mat_d.sum(axis=1, keepdims=True) / 2.0
        mat_d /= (-2.0)

        for i in range(n):
            for j in range(i, n):
                di = mat_d[:, i:i+1]
                dj = mat_d[:, j:j+1]
                if i == j:
                    mat_f[i, j] = dot(di, dj)
                else:
                    mat_f[i, j] = 2 * dot(di, dj)
        for i in range(n):
            di = mat_d[:, i:i+1]
            g[i] = 2 * dot(di, diff)
        for i in range(n):
            l += mat_f[i, i]
            mat_f[i, i] = 0
        l += norm2(diff)
        return mat_f, g, l

    def _build_qaoa_circuit(self, mat_f, g):
        """Build the QAOA circuit.

        Every rotation gate gets its own free parameter ``theta[k]``. The shared QAOA
        weights ``w = [gamma_0, beta_0, gamma_1, beta_1, ...]`` are related to them by
        ``theta = coef_matrix @ w``. This keeps each gate of the form exp(-i*theta*P/2),
        so the exact two-term parameter-shift rule applies per gate, and the gradient
        with respect to ``w`` follows from the chain rule.
        """
        n = self.n_qubit
        n_layer = self.config.n_layer
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        n_gate = n_layer * (len(pairs) + 2 * n)

        theta = ParameterVector("theta", n_gate)
        coef = np.zeros((n_gate, 2 * n_layer))

        qc = QuantumCircuit(n)
        qc.h(range(n))
        k = 0
        for layer in range(n_layer):
            # Cost layer: ZZ(gamma * f_ij) and RZ(gamma * g_i)
            for i, j in pairs:
                qc.rzz(theta[k], i, j)
                coef[k, 2 * layer] = mat_f[i, j]
                k += 1
            for i in range(n):
                qc.rz(theta[k], i)
                coef[k, 2 * layer] = g[i]
                k += 1
            qc.barrier()
            # Mixer layer: RX(beta)
            for i in range(n):
                qc.rx(theta[k], i)
                coef[k, 2 * layer + 1] = 1.0
                k += 1
            qc.barrier()
        self.coef_matrix = coef
        return qc

    def _build_qaoa_hamil(self, mat_f, g) -> SparsePauliOp:
        """Build QAOA hamiltonian: sum_i g_i Z_i + sum_{i<j} f_ij Z_i Z_j.
        """
        n = mat_f.shape[0]
        assert mat_f.shape[1] == n, "Matrix `mat_f` shape should be (n x n)"
        assert len(g) == n, "`g` length error."

        terms = [("Z", [i], float(g[i])) for i in range(n)]
        terms += [("ZZ", [i, j], float(mat_f[i, j]))
                  for i in range(n) for j in range(i + 1, n)]
        ham = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
        if self.verbose:
            print("Built hamiltonian is:\n", ham)
        return ham

    def _state(self, theta) -> Statevector:
        bound = self.circ.assign_parameters(np.asarray(theta, dtype=float))
        return Statevector.from_instruction(bound)

    def _energy(self, theta) -> float:
        return float(self._state(theta).expectation_value(self.ham).real)

    def _grad(self, w):
        """Gradient of <H> w.r.t. the shared weights `w` (parameter-shift + chain rule)."""
        theta = self.coef_matrix @ w
        grad_theta = np.zeros_like(theta)
        shift = np.pi / 2
        for k in range(len(theta)):
            theta[k] += shift
            e_plus = self._energy(theta)
            theta[k] -= 2 * shift
            e_minus = self._energy(theta)
            theta[k] += shift
            grad_theta[k] = (e_plus - e_minus) / 2.0
        return self.coef_matrix.T @ grad_theta

    # ------------------------------------------------------------------
    # Fast training path: exact NumPy statevector + analytic adjoint gradient.
    # Equivalent to the Qiskit circuit above (which is still used for sampling).
    # ------------------------------------------------------------------
    def _diag_energies(self, mat_f, g):
        """Diagonal of the hamiltonian: E(z) = sum_i g_i z_i + sum_{i<j} f_ij z_i z_j,
        z_i = +1 if bit i is 0 else -1. Basis index k has bit i = (k >> i) & 1."""
        n = self.n_qubit
        idx = np.arange(2 ** n)
        z = 1.0 - 2.0 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
        e = z @ g
        for i in range(n):
            for j in range(i + 1, n):
                e = e + mat_f[i, j] * z[:, i] * z[:, j]
        return e

    def _apply_rx_all(self, psi, beta):
        """Apply RX(beta) to every qubit. Axis a of the reshaped state is qubit n-1-a."""
        n = self.n_qubit
        c, s = np.cos(beta / 2), -1j * np.sin(beta / 2)
        psi = psi.reshape((2,) * n)
        for ax in range(n):
            psi = c * psi + s * np.flip(psi, axis=ax)
        return psi.reshape(-1)

    def _apply_x_sum(self, psi):
        """Apply sum_i X_i."""
        n = self.n_qubit
        psi = psi.reshape((2,) * n)
        out = sum(np.flip(psi, axis=ax) for ax in range(n))
        return out.reshape(-1)

    def _loss_and_grad(self, w):
        """Return <H> and d<H>/dw for w = [gamma_0, beta_0, gamma_1, beta_1, ...]."""
        n_layer = self.config.n_layer
        e = self.diag
        psi = np.full(2 ** self.n_qubit, 2 ** (-self.n_qubit / 2), dtype=complex)
        for l in range(n_layer):
            psi = np.exp(-0.5j * w[2 * l] * e) * psi
            psi = self._apply_rx_all(psi, w[2 * l + 1])
        loss = float(np.real(np.vdot(psi, e * psi)))

        grad = np.zeros_like(w)
        lam = e * psi
        for l in range(n_layer - 1, -1, -1):
            # mixer: generator G = sum X / 2, state after gate is psi
            grad[2 * l + 1] = 2 * np.real(np.vdot(lam, -0.5j * self._apply_x_sum(psi)))
            psi = self._apply_rx_all(psi, -w[2 * l + 1])
            lam = self._apply_rx_all(lam, -w[2 * l + 1])
            # cost: generator G = E / 2 (diagonal)
            grad[2 * l] = 2 * np.real(np.vdot(lam, -0.5j * e * psi))
            ph = np.exp(0.5j * w[2 * l] * e)
            psi = ph * psi
            lam = ph * lam
        return loss, grad

    def _train(self):
        """Train QAOA parameters with Adam, then sample."""
        if self.config.optimizer.lower() != "adam":
            raise ValueError("Only the 'adam' optimizer is supported.")
        t1 = time.time()
        cfg = self.config

        # MindSpore's default init was a normal distribution with sigma=0.01.
        w = self.rng.normal(0.0, 0.01, size=self.coef_matrix.shape[1])
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        loss_record = []
        for it in range(1, cfg.max_iter + 1):
            loss, grad = self._loss_and_grad(w)
            loss_record.append(loss)
            if self.verbose and (it - 1) % max(cfg.max_iter // 10, 1) == 0:
                print(f"Iter [{it - 1}]: loss = {loss:.3f}")
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** it)
            v_hat = v / (1 - beta2 ** it)
            w = w - cfg.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        t2 = time.time()
        if self.verbose:
            print(f"It spends {t2 - t1:.2f} seconds to train.")
        self.loss_record = loss_record
        self.weights = w

        sv = self._state(self.coef_matrix @ w)
        if cfg.seed is not None:
            sv.seed(cfg.seed)
        out_data = sv.sample_counts(cfg.n_sample_shot)
        if self.verbose:
            print(dict(out_data))
        # Most frequently sampled state. Qiskit bit strings are little-endian
        # (qubit 0 is the rightmost character), same as the original code assumed.
        bit_string_max = max(out_data, key=out_data.get)
        bit_float_max = [float(c) for c in bit_string_max][::-1]
        disturbance = np.array(bit_float_max) * np.array(self.symbol)
        return disturbance

    def run(self, mat_d, diff, s, bop):
        """The outer interface that get the disturbance.
        Args:
            mat_d: The LLL-reduced matrix.
            diff: The difference between target vector and the closest vector.
            s: The symbol which element is in {-1, 1}.
            bop: The closest vector from Schnorr's algorithm.
        Return:
            vnew: The new closest vector optimized based on `bop`.
        """
        self.n_qubit = mat_d.shape[1]
        mat_f, g, _ = self._get_hamil_coef(mat_d, diff, s)
        self.ham = self._build_qaoa_hamil(mat_f, g)
        self.diag = self._diag_energies(mat_f, g)
        self.circ = self._build_qaoa_circuit(mat_f, g)
        self.symbol = s
        disturbance = self._train()
        vnew = (mat_d * disturbance).sum(axis=1).reshape((-1, 1)) + bop
        return vnew

    def __call__(self, mat_d, diff, s, bop):
        return self.run(mat_d, diff, s, bop)


if __name__ == "__main__":
    mat_d = np.array(
        [[1, -4, -3],
         [-2,  1,  2],
         [2,  2,  0],
         [3, -2,  4]], np.float64)

    config = QAOAConfig(n_layer=5, max_iter=8001, verbose=1)
    qaoa = QAOASearch(config)

    diff = np.array([0, 4, 4, 2], np.float64).reshape((-1, 1))
    s = np.array([-1, -1, -1], np.float64)
    t = np.array([0, 0, 0, 240], np.float64).reshape((-1, 1))
    bop = np.array([0, 4, 4, 242], np.float64).reshape((-1, 1))
    vnew = qaoa.run(mat_d, diff, s, bop)
    print(f"target vector `t`:\n{t.ravel().tolist()}")
    print(f"schnorr's algorithm closest vector `bop`:\n{bop.ravel().tolist()}")
    print(f"new vector `vnew`:\n{vnew.ravel().tolist()}")
    print(f"norm(bop - t) = {np.linalg.norm(bop - t):.2f}")
    print(f"norm(vnew - t) = {np.linalg.norm(vnew - t):.2f}")