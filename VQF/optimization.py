"""
Qiskit port of the VQF optimization engine (previously pyQuil + Grove QAOA).
"""
import time
from functools import reduce
from itertools import product

import numpy as np
import scipy.optimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from sympy import Poly

from vqe import VQE, prepare_for_sampling, sample_bitstrings
from visualization import plot_energy_landscape, plot_variance_landscape, plot_optimization_trajectory

BASIS_GATES = ["h", "rx", "rz", "cx"]


def pass_fun(*args, **kwargs):
    """No-op used as `disp` when verbose=False."""
    pass


def _strip_identity(op: SparsePauliOp) -> SparsePauliOp:
    """Drop the identity part of an operator (it only contributes a global phase)."""
    op = op.simplify()
    keep = [i for i, p in enumerate(op.paulis) if p.x.any() or p.z.any()]
    if not keep:
        return SparsePauliOp("I" * op.num_qubits, 0.0)
    return SparsePauliOp(op.paulis[keep], op.coeffs[keep])


def make_pauli_noise_model(p):
    """
    Pauli channel with Px = Py = Pz = p applied after every gate.
    (Approximates `QVM(gate_noise=[p, p, p])`; two-qubit gates get an
    independent channel on each of their two qubits.)
    """
    err_1q = pauli_error([("X", p), ("Y", p), ("Z", p), ("I", 1 - 3 * p)])
    err_2q = err_1q.tensor(err_1q)
    noise_model = NoiseModel(basis_gates=BASIS_GATES)
    noise_model.add_all_qubit_quantum_error(err_1q, ["h", "rx", "rz"])
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])
    return noise_model


class QAOAInstance(object):
    """
    Minimal replacement for `grove.pyqaoa.qaoa.QAOA`.

    Circuit (same as Grove): H on every qubit, then for each layer
    exp(-i gamma_k * H_cost) followed by exp(-i beta_k * H_mix).
    Parameter vector layout is `[betas..., gammas...]`, as in Grove.
    """

    def __init__(self, n_qubits, steps, cost_ham, ref_ham, minimizer,
                 minimizer_kwargs=None, vqe_options=None, noise_model=None, seed=None):
        self.n_qubits = n_qubits
        self.qubits = list(range(n_qubits))
        self.steps = steps
        self.cost_ham = cost_ham          # list of SparsePauliOp
        self.ref_ham = ref_ham            # list of SparsePauliOp
        self.minimizer = minimizer
        self.minimizer_args = []
        self.minimizer_kwargs = minimizer_kwargs or {}
        self.vqe_options = vqe_options or {}
        self.betas = np.array([])
        self.gammas = np.array([])

        self.noise_model = noise_model
        backend_kwargs = {"seed_simulator": seed} if seed is not None else {}
        self.backend = (AerSimulator(noise_model=noise_model, **backend_kwargs) if noise_model
                        else AerSimulator(**backend_kwargs))

        self._cost_sum = reduce(lambda a, b: a + b, cost_ham).simplify()
        self._cost_evolution_op = _strip_identity(self._cost_sum)
        self._mixer_sum = reduce(lambda a, b: a + b, ref_ham).simplify()
        self._circuit_cache = {}
        self._prepared_cache = {}

    @property
    def total_cost_hamiltonian(self):
        return self._cost_sum

    def get_parameterized_circuit(self, steps=None):
        """Parameterized QAOA circuit with `steps` layers (cached)."""
        steps = self.steps if steps is None else steps
        if steps not in self._circuit_cache:
            betas = ParameterVector("beta", steps)
            gammas = ParameterVector("gamma", steps)
            qc = QuantumCircuit(self.n_qubits)
            qc.h(range(self.n_qubits))
            for k in range(steps):
                qc.append(PauliEvolutionGate(self._cost_evolution_op, time=gammas[k]),
                          range(self.n_qubits))
                qc.append(PauliEvolutionGate(self._mixer_sum, time=betas[k]),
                          range(self.n_qubits))
            # Synthesize PauliEvolutionGate into basic gates. Otherwise StatevectorEstimator
            # evaluates it by matrix exponentiation on every call (~45x slower, identical
            # results to ~1e-13), and gate counts reflect the real circuit.
            qc = qc.decompose(reps=3)
            # Parameter order must be [betas..., gammas...]
            names = [p.name for p in qc.parameters]
            assert names == [f"beta[{i}]" for i in range(steps)] + \
                            [f"gamma[{i}]" for i in range(steps)], names
            self._circuit_cache[steps] = qc
        return self._circuit_cache[steps]

    def get_prepared_circuit(self, steps=None):
        """Measured + transpiled circuit for shot-based runs (cached)."""
        steps = self.steps if steps is None else steps
        if steps not in self._prepared_cache:
            self._prepared_cache[steps] = prepare_for_sampling(
                self.get_parameterized_circuit(steps), self.backend)
        return self._prepared_cache[steps]

    def get_string(self, betas, gammas, samples=1000):
        """
        Sample the QAOA state. Returns (most_frequent_bitstring, Counter), where
        bitstrings are tuples with index i = qubit i (same as Grove).
        """
        params = np.hstack((betas, gammas))
        steps = len(betas)
        freq = sample_bitstrings(self.get_prepared_circuit(steps), params,
                                 self.backend, samples)
        most_frequent = max(freq, key=lambda x: freq[x])
        return most_frequent, freq


class OptimizationEngine(object):
    """
    The optimization engine for the VQF algorithm.

    This class takes a problem encoded as clauses, further encodes it into a Hamiltonian
    and solves it using QAOA.

    Args:
        clauses (list): List of clauses (sympy expressions) representing the problem.
        m (int): Number to be factored. Needed only for the purpose of tagging result files.
        steps (int, optional): Number of steps in the QAOA algorithm. Default: 1
        grid_size (int, optional): The resolution of the grid for grid search. Default: None
        tol (float, optional): Parameter of BFGS optimization method. Gradient norm must be less than tol before successful termination. Default:1e-5
        gate_noise (float, optional): Pauli-channel gate noise probability (Px=Py=Pz). Default: None.
            When set, evaluations are shot-based and COBYLA replaces BFGS (BFGS's finite-difference
            gradient is pure noise on a shot-noisy objective).
        seed (int, optional): Seeds numpy and the Aer simulator for reproducible runs. Default: None
        verbose (bool): If True, information about the execution will be printed to the console. Default: False
        visualize (bool): Flag indicating if visualizations should be created. Default: False

    Attributes:
        clauses (list): See Args.
        grid_size (int): See Args.
        mapping (dict): Maps variables into qubit indices.
        qaoa_inst (QAOAInstance): Replacement for Grove's QAOA class.
        samples (int): If noise model is active, how many shots to take per circuit evaluation.
        ax (object): Matplotlib `axis` object, used for plotting optimization trajectory.
    """
    def __init__(self, clauses, m=None, steps=1, grid_size=None, tol=1e-5, gate_noise=None,
                 verbose=False, visualize=False, seed=None):
        self.clauses = clauses
        self.m = m
        self.verbose = verbose
        self.visualize = visualize
        self.gate_noise = gate_noise

        self.mapping = self._build_mapping()
        self.n_qubits = len(self.mapping)
        if self.n_qubits == 0:
            raise ValueError("No unknowns left after preprocessing: nothing to run on a quantum "
                             "device (all clauses are constant).")
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        # resource counters (see resource_report)
        self.n_grid_evals = 0
        self.n_opt_evals = 0
        self.final_shots = 10000
        self._last_grid = None

        # (original code referenced `qubits` here before it was defined)
        if grid_size is None:
            self.grid_size = len(clauses) + self.n_qubits
        else:
            self.grid_size = grid_size

        cost_operators = self.create_operators_from_clauses()
        mixing_operators = self.create_mixing_operators()
        print_fun = print if self.verbose else pass_fun

        if gate_noise:
            self.samples = int(1e3)
            noise_model = make_pauli_noise_model(gate_noise)
            # gradient-free: finite differences (eps~1e-8) on a 1000-shot objective are pure noise
            minimizer_kwargs = {'method': 'COBYLA',
                                'options': {'rhobeg': 0.2, 'tol': 1e-3, 'maxiter': 200}}
        else:
            self.samples = None
            noise_model = None
            minimizer_kwargs = {'method': 'BFGS',
                                'options': {'gtol': tol, 'disp': False}}

        vqe_option = {'disp': print_fun, 'return_all': True, 'samples': self.samples}

        self.qaoa_inst = QAOAInstance(self.n_qubits,
                                      steps=steps,
                                      cost_ham=cost_operators,
                                      ref_ham=mixing_operators,
                                      minimizer=scipy.optimize.minimize,
                                      minimizer_kwargs=minimizer_kwargs,
                                      vqe_options=vqe_option,
                                      noise_model=noise_model,
                                      seed=seed)
        self.ax = None

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------
    def _build_mapping(self):
        """Maps variable names to qubit indices, in order of first appearance."""
        mapping = {}
        for clause in self.clauses:
            if clause == 0:
                continue
            for variable in sorted(clause.free_symbols, key=str):
                if str(variable) not in mapping:
                    mapping[str(variable)] = len(mapping)
        return mapping

    def create_operators_from_clauses(self):
        """
        Creates the cost Hamiltonian from clauses: for each clause C, the
        operator C(x -> (I - Z)/2) squared. For details see section IIC of the article.

        Instead of special-casing constant/linear/bilinear terms like the old
        code, each clause is expanded as a polynomial in the binary variables
        (x^k = x), so terms of any order are handled.
        """
        n = self.n_qubits
        identity = SparsePauliOp("I" * n)
        # binary variable x_i = (I - Z_i) / 2
        x_ops = {
            name: 0.5 * identity - 0.5 * SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], n)
            for name, q in self.mapping.items()
        }

        operators = []
        for clause in self.clauses:
            if clause == 0:
                continue
            symbols = sorted(clause.free_symbols, key=str)
            clause_operator = SparsePauliOp("I" * n, 0.0)
            if not symbols:
                clause_operator = clause_operator + float(clause) * identity
            else:
                for exponents, coeff in Poly(clause, *symbols).terms():
                    term = float(coeff) * identity
                    for symbol, exp in zip(symbols, exponents):
                        if exp > 0:  # binary variables: x**k == x
                            term = term.compose(x_ops[str(symbol)])
                    clause_operator = clause_operator + term
            clause_operator = clause_operator.simplify()
            squared_clause_operator = clause_operator.compose(clause_operator).simplify()
            if self.verbose:
                print("C:", clause_operator)
                print("C**2:", squared_clause_operator)
            operators.append(squared_clause_operator)
        return operators

    def create_mixing_operators(self):
        """
        Creates mixing Hamiltonian. (eq. 10)
        """
        return [SparsePauliOp.from_sparse_list([("X", [q], -1.0)], self.n_qubits)
                for q in self.mapping.values()]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _energies(self, param_matrix):
        """Energies for each row of `param_matrix` (columns: betas..., gammas...)."""
        circuit = self.qaoa_inst.get_parameterized_circuit()
        hamiltonian = self.qaoa_inst.total_cost_hamiltonian
        self.n_grid_evals += len(param_matrix)
        if self.samples is None:
            return VQE.expectation_batch(circuit, hamiltonian, param_matrix)
        prepared = self.qaoa_inst.get_prepared_circuit()
        return np.array([
            VQE.expectation(circuit, hamiltonian, row, self.samples,
                            self.qaoa_inst.backend, prepared)
            for row in param_matrix])

    def resource_report(self):
        """Resources used, for like-for-like comparison with other methods.

        Call after perform_qaoa(). In the noiseless case grid/optimizer evaluations are exact
        statevector evaluations (0 shots); with gate noise each costs `samples` shots.
        """
        prepared = self.qaoa_inst.get_prepared_circuit()
        ops = prepared.count_ops()
        shots_per_eval = self.samples or 0
        return {
            "n_qubits": self.n_qubits,
            "steps": self.qaoa_inst.steps,
            "cx_count": int(ops.get("cx", 0)),
            "depth": int(prepared.depth()),
            "grid_evals": int(self.n_grid_evals),
            "optimizer_evals": int(self.n_opt_evals),
            "final_shots": int(self.final_shots),
            "shots_per_eval": shots_per_eval,
            "total_circuit_evals": int(self.n_grid_evals + self.n_opt_evals + 1),
            "total_shots": int((self.n_grid_evals + self.n_opt_evals) * shots_per_eval + self.final_shots),
        }

    # ------------------------------------------------------------------
    # QAOA driver
    # ------------------------------------------------------------------
    def perform_qaoa(self):
        """
        Finds optimal angles for QAOA.

        Returns:
            sampling_results (Counter): Counter, where each element represents a bitstring that has been obtained.
            mapping (dict): See class description.
        """
        betas, gammas = self.step_by_step_grid_search_angles()
        if self.visualize and self.qaoa_inst.steps == 1 and self._last_grid is not None:
            grid_betas, grid_gammas, grid_energies = self._last_grid
            self.ax = plot_energy_landscape(grid_betas, grid_gammas, grid_energies, log_legend=True)
        self.qaoa_inst.betas = betas
        self.qaoa_inst.gammas = gammas
        betas, gammas = self.get_angles()
        _, sampling_results = self.qaoa_inst.get_string(betas, gammas, samples=self.final_shots)
        return sampling_results, self.mapping

    def get_angles(self):
        """
        Finds optimal angles with the variational eigensolver method (BFGS refinement
        starting from the grid-search angles).

        Returns:
            best_betas, best_gammas (np.arrays): best values of the betas and gammas found.
        """
        q = self.qaoa_inst
        stacked_params = np.hstack((q.betas, q.gammas))
        vqe = VQE(q.minimizer, minimizer_args=q.minimizer_args,
                  minimizer_kwargs=q.minimizer_kwargs)
        result = vqe.vqe_run(q.get_parameterized_circuit(), q.total_cost_hamiltonian,
                             stacked_params,
                             backend=q.backend if self.samples else None,
                             **q.vqe_options)
        self.n_opt_evals = vqe.number_of_evaluations
        best_betas = result.x[:q.steps]
        best_gammas = result.x[q.steps:]
        optimization_trajectory = result.iteration_params
        energy_history = result.expectation_vals  # noqa: F841 (kept for inspection)

        if self.ax is not None and self.visualize and q.steps == 1:
            plot_optimization_trajectory(self.ax, optimization_trajectory)
        return best_betas, best_gammas

    def simple_grid_search_angles(self, save_data=False):
        """
        Finds optimal angles for QAOA by performing grid search on all the angles.
        This is not recommended for higher values of steps parameter,
        since it results in grid_size**(2*steps) evaluations.

        Returns:
            best_betas, best_gammas (np.arrays): best values of the betas and gammas found.
        """
        steps = self.qaoa_inst.steps

        # For some reasons np.meshgrid returns columns in order, where values in second
        # grow slower than in the first one. This a fix to it.
        if steps == 1:
            column_order = [0]
        else:
            column_order = [1, 0] + list(range(2, steps))

        beta_ranges = [np.linspace(0, np.pi, self.grid_size)] * steps
        all_betas = np.vstack(np.meshgrid(*beta_ranges)).reshape(steps, -1).T
        all_betas = all_betas[:, column_order]

        gamma_ranges = [np.linspace(0, 2 * np.pi, self.grid_size)] * steps
        all_gammas = np.vstack(np.meshgrid(*gamma_ranges)).reshape(steps, -1).T
        all_gammas = all_gammas[:, column_order]

        # same ordering as the old nested loop: betas outer, gammas inner
        pairs = list(product(range(len(all_betas)), range(len(all_gammas))))
        param_matrix = np.array([np.hstack((all_betas[i], all_gammas[j])) for i, j in pairs])
        all_energies = self._energies(param_matrix)

        best = int(np.argmin(all_energies))
        best_betas = all_betas[pairs[best][0]]
        best_gammas = all_gammas[pairs[best][1]]
        if self.verbose:
            print("Lowest energy:", all_energies[best])
            print("Angles:", best_betas, best_gammas)

        if save_data:
            file_name = "_".join([str(self.m), "grid", str(self.grid_size), str(time.time())]) + ".csv"
            data = np.hstack([param_matrix, all_energies[:, None]])
            np.savetxt(file_name, data, delimiter=",")

        if self.visualize:
            if steps == 1:
                self.ax = plot_energy_landscape(all_betas, all_gammas, np.array(all_energies),
                                                log_legend=True)
            else:
                plot_variance_landscape(all_betas, all_gammas, np.array(all_energies))

        return best_betas, best_gammas

    def step_by_step_grid_search_angles(self):
        """
        Finds optimal angles by "step-by-step" grid search: optimize the first
        (beta, gamma) pair, fix it, then optimize the second pair, and so on.
        Requires steps*grid_size**2 evaluations.

        Returns:
            best_betas, best_gammas (np.arrays): best values of the betas and gammas found.
        """
        max_step = self.qaoa_inst.steps
        original_steps = self.qaoa_inst.steps
        self.qaoa_inst.betas = np.array([])
        self.qaoa_inst.gammas = np.array([])
        best_betas = np.array([])
        best_gammas = np.array([])
        try:
            for current_step in range(1, max_step + 1):
                if self.verbose:
                    print("step:", current_step, "\n")
                beta, gamma = self.one_step_grid_search(current_step)
                best_betas = np.append(best_betas, beta)
                best_gammas = np.append(best_gammas, gamma)
                self.qaoa_inst.betas = best_betas
                self.qaoa_inst.gammas = best_gammas
        finally:
            self.qaoa_inst.steps = original_steps
        return best_betas, best_gammas

    def one_step_grid_search(self, current_step):
        """
        Grid search on n-th pair of QAOA angles, where n=current_step.

        Args:
            current_step (int): specify on which layer do we perform search.

        Returns:
            best_beta, best_gamma (floats): best values of the beta and gamma found.
        """
        self.qaoa_inst.steps = current_step
        fixed_betas = self.qaoa_inst.betas
        fixed_gammas = self.qaoa_inst.gammas
        beta_range = np.linspace(0, np.pi, self.grid_size)
        gamma_range = np.linspace(0, 2 * np.pi, self.grid_size)

        grid = list(product(beta_range, gamma_range))   # beta outer, gamma inner
        param_matrix = np.array([
            np.hstack((np.append(fixed_betas, b), np.append(fixed_gammas, g)))
            for b, g in grid])
        energies = self._energies(param_matrix)
        self._last_grid = (beta_range, gamma_range, energies)
        best = int(np.argmin(energies))
        return grid[best]