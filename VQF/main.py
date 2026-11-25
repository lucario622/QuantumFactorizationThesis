from preprocessing import create_clauses, calculate_number_of_unknowns
from preprocessing import factor_56153, factor_291311
from optimization import OptimizationEngine
from sympy import Add, Mul, Symbol
import pdb
import time


def build_problem(m, true_p=None, true_q=None, use_true_values=False,
                  apply_preprocessing=True, preprocessing_verbose=False):
    """Returns p_dict, q_dict, z_dict, clauses. If use_true_values, the *bit lengths* of the
    true factors are given to the solver (as in the VQF paper); otherwise nothing is leaked."""
    if m == 56153:
        return factor_56153()
    if m == 291311:
        return factor_291311()
    if use_true_values:
        return create_clauses(m, true_p, true_q, apply_preprocessing, preprocessing_verbose)
    return create_clauses(m, None, None, apply_preprocessing, preprocessing_verbose)


def evaluate_success(sampling_results, mapping, p_dict, q_dict, m, top_k=10):
    """End-to-end, ground-truth-free success metrics (only m is needed).

    A bitstring "succeeds" if the (p, q) it decodes to are non-trivial and p*q == m.

    Returns dict with:
        top1_success      : the most frequent bitstring decodes to a valid factorisation
        topk_success      : any of the top_k most frequent bitstrings does
        shot_success_prob : fraction of shots that decode to a valid factorisation
        factors           : factor pair from the most frequent successful bitstring (or None)
    """
    total = sum(sampling_results.values())
    shot_success = 0
    top1 = False
    topk = False
    factors = None
    for rank, (bit_string, count) in enumerate(sampling_results.most_common()):
        # update_dictionary mutates its argument -> pass copies
        try:
            p, q = decode_solution(update_dictionary(bit_string, mapping, dict(p_dict)),
                                   update_dictionary(bit_string, mapping, dict(q_dict)))
            p, q = int(p), int(q)
        except (TypeError, ValueError):
            continue
        ok = p > 1 and q > 1 and p * q == m
        if ok:
            shot_success += count
            if factors is None:
                factors = (p, q)
            if rank == 0:
                top1 = True
            if rank < top_k:
                topk = True
    return {
        "top1_success": top1,
        "topk_success": topk,
        "shot_success_prob": shot_success / total if total else 0.0,
        "factors": factors,
    }


def factor_number(m, true_p=None, true_q=None, use_true_values=False, seed=None,
                  steps=1, grid_size=20, gate_noise=None, compute_overlap=True,
                  visualize=False):
    """Runs the VQF pipeline. Returns (p, q, squared_overlap, report), where `report`
    holds the success metrics (evaluate_success) and the resource report."""
    optimization_verbose = False
    p_dict, q_dict, z_dict, clauses = build_problem(m, true_p, true_q, use_true_values)

    number_of_uknowns, number_of_carry_bits = calculate_number_of_unknowns(
        p_dict, q_dict, z_dict
    )
    print("Number of unknowns:", number_of_uknowns)
    print("Number of carry bits:", number_of_carry_bits)

    optimization_engine = OptimizationEngine(
        clauses,
        m,
        steps=steps,
        grid_size=grid_size,
        gate_noise=gate_noise,
        verbose=optimization_verbose,
        visualize=visualize,
        seed=seed,
    )
    assert (
        optimization_engine.n_qubits > 0
    ), "Expected at least 1 qubit to always run the quantum step"
    print(f"Running quantum QAOA step on {optimization_engine.n_qubits} qubits")
    sampling_results, mapping = optimization_engine.perform_qaoa()
    most_frequent_bit_string = max(sampling_results, key=lambda x: sampling_results[x])

    report = evaluate_success(sampling_results, mapping, p_dict, q_dict, m)
    report.update(optimization_engine.resource_report())

    squared_overlap = None
    if compute_overlap and true_p is not None and true_q is not None:
        # Legacy diagnostic: needs the true factors (and hard-coded symmetric cases).
        squared_overlap = calculate_squared_overlap(
            mapping, sampling_results, true_p, true_q, p_dict, q_dict
        )
    p_dict = update_dictionary(most_frequent_bit_string, mapping, p_dict)
    q_dict = update_dictionary(most_frequent_bit_string, mapping, q_dict)
    z_dict = update_dictionary(most_frequent_bit_string, mapping, z_dict)

    p, q = decode_solution(p_dict, q_dict)
    return p, q, squared_overlap, report


def calculate_squared_overlap(
    mapping, sampling_results, true_p, true_q, p_dict, q_dict
):
    """LEGACY diagnostic; needs the true factors and hard-coded symmetric cases.
    Prefer evaluate_success() for method comparison."""
    p_binary_string = bin(true_p)[2:][::-1]
    q_binary_string = bin(true_q)[2:][::-1]

    p_binary = [int(char) for char in p_binary_string]
    q_binary = [int(char) for char in q_binary_string]
    if len(p_binary) < len(p_dict):
        trailing_zeros = len(p_dict) - len(p_binary)
        for zero in range(trailing_zeros):
            p_binary.append(0)

    if len(q_binary) < len(q_dict):
        trailing_zeros = len(q_dict) - len(q_binary)
        for zero in range(trailing_zeros):
            q_binary.append(0)

    all_correct_assignments = []
    correct_assignment = {}
    for q_id, q_val in q_dict.items():
        if type(q_val) is Symbol:
            bit_id = mapping[str(q_val)]
            correct_value = q_binary[q_id]
            if bit_id not in correct_assignment.keys():
                correct_assignment[bit_id] = correct_value

    for p_id, p_val in p_dict.items():
        if type(p_val) is Symbol:
            bit_id = mapping[str(p_val)]
            correct_value = p_binary[p_id]
            if bit_id not in correct_assignment.keys():
                correct_assignment[bit_id] = correct_value

    all_correct_assignments.append(correct_assignment)
    # TODO:
    # This is just a hack for 56153 and 291311 to work properly.
    # It should be generalized to work for any symmetric case.

    if true_p == 241 and true_q == 233:
        assignment_1 = {
            mapping["p_3"]: 0,
            mapping["p_4"]: 1,
            mapping["q_3"]: 1,
            mapping["q_4"]: 0,
        }
        assignment_2 = {
            mapping["p_3"]: 1,
            mapping["p_4"]: 0,
            mapping["q_3"]: 0,
            mapping["q_4"]: 1,
        }
        all_correct_assignments = [assignment_1, assignment_2]

    if true_p == 557 and true_q == 523:
        assignment_1 = {
            mapping["p_1"]: 0,
            mapping["p_2"]: 1,
            mapping["p_5"]: 1,
            mapping["q_1"]: 1,
            mapping["q_2"]: 0,
            mapping["q_5"]: 0,
        }
        assignment_2 = {
            mapping["p_1"]: 1,
            mapping["p_2"]: 0,
            mapping["p_5"]: 0,
            mapping["q_1"]: 0,
            mapping["q_2"]: 1,
            mapping["q_5"]: 1,
        }
        all_correct_assignments = [assignment_1, assignment_2]

    total_overlap = 0
    total_count = 0
    print("Correct assignments:")
    print(all_correct_assignments)
    print("Variables mapping")
    print(mapping)
    squared_overlap = 0
    print("Sample, count, correct, sq_overlap")
    for correct_assignment in all_correct_assignments:
        total_overlap = 0
        total_count = 0
        for bit_string, count in sampling_results.most_common():
            correct_count = 0
            for bit_id, bit_value in enumerate(bit_string):
                # This accounts for the fact some of the bits of the sampling results
                # are irrelevant to the result - namely, carry bits.
                if bit_id not in correct_assignment.keys():
                    continue
                if bit_value == correct_assignment[bit_id]:
                    correct_count += 1
            overlap = (correct_count / len(correct_assignment)) ** 2 * count
            total_count += count
            print(bit_string, count, correct_count, overlap)
            total_overlap += overlap
        print("_" * 10)
        if total_count > 0:
            total_overlap = total_overlap / total_count
        squared_overlap += total_overlap
    return squared_overlap


def update_dictionary(qaoa_solution, mapping, x_dict):
    # values_dict = {symbol_str: qaoa_solution[index] for symbol_str, index in mapping.items()}
    symbols_dict = {
        Symbol(symbol_str): qaoa_solution[index]
        for symbol_str, index in mapping.items()
    }
    for key, value in x_dict.items():
        if value in symbols_dict.keys():
            x_dict[key] = symbols_dict[value]
        if type(value) == Add or type(value) == Mul:
            x_dict[key] = value.subs(symbols_dict)
    return x_dict


def decode_solution(p_dict, q_dict):
    p = 0
    for key, value in p_dict.items():
        p += value * 2**key

    q = 0
    for key, value in q_dict.items():
        q += value * 2**key

    return p, q


def main():
    true_p = 557
    true_q = 523
    m = 291311
    use_true_values = False

    t_0 = time.time()
    print("M:", m)
    p, q, squared_overlap, report = factor_number(
        m, true_p, true_q, use_true_values=use_true_values, seed=0
    )
    t_1 = time.time()
    print(f"Time taken in seconds: {t_1 - t_0:.3f}")
    print("Calculated primes of ", m, "are:", p, "and", q)
    print("      True primes of ", m, "are:", true_p, "and", true_q)
    print("Squared overlap (legacy diagnostic):", squared_overlap)
    print("Success report:", report)
    # p_q_m_list = [
    #     [283, 7, 1981],
    #     [29, 11, 319],
    #     [263, 263, 69169],
    #     [263, 11, 2893],
    #     [241, 233, 56153],
    #     [557, 523, 291311],
    # ]


if __name__ == "__main__":
    main()