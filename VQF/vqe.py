"""
Qiskit port of the (Grove/pyQuil) VQE helper.

What changed relative to the pyQuil version
-------------------------------------------
* The ansatz is now a *parameterized* ``qiskit.QuantumCircuit`` rather than a
  Python function returning a pyQuil ``Program``.
* The Hamiltonian is a ``qiskit.quantum_info.SparsePauliOp`` rather than a
  pyQuil ``PauliSum``.
* ``samples=None``  -> exact expectation via ``StatevectorEstimator``
  (replaces ``WavefunctionSimulator().expectation``).
* ``samples=int``   -> shot-based estimate on an Aer backend (optionally noisy)
  (replaces ``QVM(gate_noise=...)`` + ``expectation_from_sampling``).
* No ``QuantumComputer`` / ``QVM`` / ``funcsigs`` any more.
"""

import inspect
from collections import Counter
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


class OptResults(dict):
    """
    Object for holding optimization results from VQE.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------
def prepare_for_sampling(circuit: QuantumCircuit, backend=None) -> QuantumCircuit:
    """
    Append measurements and transpile once for ``backend``.

    The returned circuit still has free parameters, so it can be re-bound and
    re-run many times without paying the transpilation cost again.
    """
    backend = backend if backend is not None else AerSimulator()
    measured = circuit.copy()
    measured.measure_all()
    return transpile(measured, backend, optimization_level=1)


def sample_bitstrings(
    prepared: QuantumCircuit, params: Sequence[float], backend, shots: int
) -> Counter:
    """
    Bind ``params`` into a circuit from :func:`prepare_for_sampling`, run it and
    return a ``Counter`` whose keys are tuples of ints with **index i = qubit i**
    (the same convention the pyQuil/Grove code used).

    Qiskit's own count strings are little-endian (qubit 0 is the right-most
    character), so they are reversed here.
    """
    bound = prepared.assign_parameters(np.asarray(params, dtype=float).tolist())
    counts = backend.run(bound, shots=shots).result().get_counts()
    freq = Counter()
    for bitstring, n in counts.items():
        key = tuple(int(b) for b in reversed(bitstring.replace(" ", "")))
        freq[key] += n
    return freq


def energy_from_samples(hamiltonian: SparsePauliOp, freq: Counter) -> float:
    """
    Estimate <H> from computational-basis samples for a Hamiltonian that is
    diagonal (only I and Z terms), which is the case for the VQF cost operator.
    This replaces the old parity-based ``expectation_from_sampling``.
    """
    if hamiltonian.paulis.x.any():
        raise NotImplementedError(
            "Sample-based energy only supports I/Z-only (diagonal) Hamiltonians."
        )
    shots = sum(freq.values())
    bits = np.array(list(freq.keys()), dtype=int)  # (n_strings, n_qubits)
    weights = np.array(list(freq.values()), dtype=float) / shots
    z = hamiltonian.paulis.z.astype(int)  # (n_terms, n_qubits)
    signs = 1 - 2 * ((bits @ z.T) % 2)  # (n_strings, n_terms)
    per_string = signs @ hamiltonian.coeffs  # (n_strings,)
    return float(np.real(weights @ per_string))


# --------------------------------------------------------------------------
# VQE
# --------------------------------------------------------------------------
class VQE(object):
    """
    The Variational-Quantum-Eigensolver algorithm.

    1) initialize with ``inst = VQE(minimizer)`` where ``minimizer`` is e.g.
       ``scipy.optimize.minimize``.
    2) call ``inst.vqe_run(ansatz, hamiltonian, initial_parameters, ...)``.

    :param minimizer: function that minimizes an objective, called as
        ``minimizer(objective, initial_params, *minimizer_args, **minimizer_kwargs)``.
    :param minimizer_args: (list) extra positional arguments for the minimizer.
    :param minimizer_kwargs: (dict) keyword arguments for the minimizer.
    """

    def __init__(self, minimizer, minimizer_args=None, minimizer_kwargs=None):
        self.minimizer = minimizer
        self.minimizer_args = list(minimizer_args) if minimizer_args else []
        # copy: we add 'callback' / 'jac' below and must not mutate the caller's dict
        self.minimizer_kwargs = dict(minimizer_kwargs) if minimizer_kwargs else {}
        self.n_qubits = None

    def vqe_run(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        initial_params,
        backend=None,
        jacobian=None,
        disp=None,
        samples: Optional[int] = None,
        return_all=False,
    ):
        """
        Functional minimization loop.

        :param ansatz: parameterized ``QuantumCircuit`` (no measurements).
            Parameters are bound in the order of ``ansatz.parameters``.
        :param hamiltonian: ``SparsePauliOp`` whose expectation is minimized.
        :param initial_params: (ndarray) initial parameter vector.
        :param backend: (optional) Aer backend, e.g. one built with a noise
            model. Only used when ``samples`` is not None.
        :param jacobian: (optional) gradient function.
        :param disp: (optional, callable) called with strings to report progress.
        :param samples: (int) shots per expectation value. ``None`` -> exact
            statevector evaluation (noise models are then ineffective).
        :param return_all: (bool) also return all intermediate parameters and
            expectation values.
        :return: ``OptResults`` with ``x``, ``fun`` and, if requested,
            ``iteration_params`` and ``expectation_vals``.
        """
        self.number_of_evaluations = 0
        self._disp_fun = disp if disp is not None else (lambda x: None)
        iteration_params = []
        expectation_vals = []
        self._current_expectation = None
        eval_cache = {}  # params bytes -> energy, so callbacks report E at iter_vars

        if samples is None and backend is not None:
            print(
                "WARNING: exact (statevector) expectation will be used; "
                "noise models will be ineffective. Pass samples=<int> to sample."
            )

        prepared = None
        if samples is not None:
            backend = backend if backend is not None else AerSimulator()
            prepared = prepare_for_sampling(ansatz, backend)

        def objective_function(params):
            mean_value = self.expectation(
                ansatz, hamiltonian, params, samples, backend, prepared
            )
            self._current_expectation = mean_value  # store for printing
            eval_cache[np.asarray(params, dtype=float).tobytes()] = mean_value
            self.number_of_evaluations += 1
            return mean_value

        def print_current_iter(iter_vars):
            # The last objective call is often a finite-difference/line-search probe, not
            # the energy at iter_vars. Look it up by parameters (re-evaluate if absent).
            energy = eval_cache.get(np.asarray(iter_vars, dtype=float).tobytes())
            if energy is None:
                energy = self.expectation(
                    ansatz, hamiltonian, iter_vars, samples, backend, prepared
                )
            self._current_expectation = energy
            self._disp_fun("\tParameters: {} ".format(iter_vars))
            if jacobian is not None:
                grad = jacobian(iter_vars)
                self._disp_fun("\tGrad-L1-Norm: {}".format(np.max(np.abs(grad))))
                self._disp_fun("\tGrad-L2-Norm: {} ".format(np.linalg.norm(grad)))
            self._disp_fun("\tE => {}".format(self._current_expectation))
            if return_all:
                iteration_params.append(np.array(iter_vars))
                expectation_vals.append(self._current_expectation)

        arguments = inspect.signature(self.minimizer).parameters.keys()
        if (disp is not None or return_all) and "callback" in arguments:
            self.minimizer_kwargs["callback"] = print_current_iter

        args = [objective_function, initial_params]
        args.extend(self.minimizer_args)
        if "jac" in arguments:
            self.minimizer_kwargs["jac"] = jacobian

        result = self.minimizer(*args, **self.minimizer_kwargs)

        if hasattr(result, "status") and result.status != 0:
            self._disp_fun(
                "Classical optimization exited with an error index: %i" % result.status
            )

        results = OptResults()
        results.nfev = self.number_of_evaluations
        if hasattr(result, "x"):
            results.x = result.x
            results.fun = result.fun
        else:
            results.x = result

        if return_all:
            results.iteration_params = iteration_params
            results.expectation_vals = expectation_vals
        return results

    @staticmethod
    def expectation(
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params,
        samples: Optional[int] = None,
        backend=None,
        prepared: Optional[QuantumCircuit] = None,
    ) -> float:
        """
        Expectation value of ``hamiltonian`` in the state ``ansatz(params)``.

        :param ansatz: parameterized circuit without measurements.
        :param hamiltonian: ``SparsePauliOp``.
        :param params: parameter values, in the order of ``ansatz.parameters``.
        :param samples: shots; ``None`` -> exact statevector evaluation.
        :param backend: Aer backend used when sampling (can carry a noise model).
        :param prepared: optional output of :func:`prepare_for_sampling`, to
            avoid re-transpiling on every call.
        """
        if not isinstance(hamiltonian, SparsePauliOp):
            raise TypeError("hamiltonian must be a qiskit SparsePauliOp")

        if samples is None:
            pub = (ansatz, hamiltonian, np.asarray(params, dtype=float))
            evs = StatevectorEstimator().run([pub]).result()[0].data.evs
            return float(np.real(evs))

        if not isinstance(samples, int):
            raise TypeError("samples variable must be an integer")
        if samples <= 0:
            raise ValueError("samples variable must be a positive integer")

        backend = backend if backend is not None else AerSimulator()
        if prepared is None:
            prepared = prepare_for_sampling(ansatz, backend)
        freq = sample_bitstrings(prepared, params, backend, samples)
        return energy_from_samples(hamiltonian, freq)

    @staticmethod
    def expectation_batch(
        ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, param_matrix
    ) -> np.ndarray:
        """
        Exact expectation values for many parameter vectors at once
        (shape ``(N, n_params)`` -> ``(N,)``). Much faster than looping.
        """
        pub = (ansatz, hamiltonian, np.asarray(param_matrix, dtype=float))
        return np.real(StatevectorEstimator().run([pub]).result()[0].data.evs)
