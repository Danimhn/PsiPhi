import math

import qiskit as q
from qiskit import ClassicalRegister
from scipy.integrate import quad
from sympy import chebyshevt

from InstructionSet import *
import numpy as np
from SubSpace import *
from qiskit.circuit import Qubit
from math import sqrt


# Apply ref(computational_basis) = (I - 2|computational_basis><computational_basis|)
def reflect_abt_computational_basis(instruction_set: InstructionSet, computational_basis: [int],
                                    qargs: [Qubit], ancilla):
    if len(computational_basis) == 0:
        raise Exception("Empty computational basis provided")
    if len(computational_basis) != len(qargs):
        raise Exception("Length of computational basis doesn't match length of qargs provided")
    use_as_control = []
    for i, value in enumerate(computational_basis):
        if value == 0:
            instruction_set.add_instruction("x", qargs[i], [])
            use_as_control.append(i)
        elif value == 1:
            use_as_control.append(i)

    if len(use_as_control) == 0:
        raise Exception("No concrete entries in the computational basis")
    elif len(use_as_control) == 1:
        instruction_set.add_instruction("z", qargs[use_as_control[0]], [])
    else:
        instruction_set.multiControl("z", [qargs[i] for i in use_as_control[:len(use_as_control) - 1]],
                                     qargs[use_as_control[len(use_as_control) - 1]], [])
    for i, value in enumerate(computational_basis):
        if value == 0:
            instruction_set.add_instruction("x", qargs[i], [])


def partial_reflect_abt_computational_basis(instruction_set: InstructionSet, computational_basis: [int], angle,
                                            qargs: [Qubit], ancilla):
    if len(computational_basis) == 0:
        raise Exception("Empty computational basis provided")
    if len(computational_basis) != len(qargs):
        raise Exception("Length of computational basis doesn't match length of qargs provided")
    use_as_control = []
    for i, value in enumerate(computational_basis):
        if value == 0:
            instruction_set.add_instruction("x", [qargs[i]], [])
            use_as_control.append(i)
        elif value == 1:
            use_as_control.append(i)

    if len(use_as_control) == 0:
        raise Exception("No concrete entries in the computational basis")
    elif len(use_as_control) == 1:
        instruction_set.add_instruction("rz", qargs[use_as_control[0]], [angle])
    else:
        instruction_set.multiControl("p", [qargs[i] for i in use_as_control[:len(use_as_control) - 1]],
                                     qargs[use_as_control[len(use_as_control) - 1]], [angle])
    for i, value in enumerate(computational_basis):
        if value == 0:
            instruction_set.add_instruction("x", [qargs[i]], [])


# Apply ref(computational_basis) = (I - 2|computational_basis><computational_basis|) for all comp bases in states.
def reflect_abt_computational_subspace(instruction_set: InstructionSet, sub_space: SubSpace, qargs: [Qubit], ancilla):
    for basis in sub_space.get_basis():
        reflect_abt_computational_basis(instruction_set, basis, qargs, ancilla)


def partial_reflect_abt_subspace(instruction_set: InstructionSet, sub_space: SubSpace, angle, qargs: [Qubit], ancilla):
    for basis in sub_space.get_basis():
        partial_reflect_abt_computational_basis(instruction_set, basis, angle, qargs, ancilla)


def amplitude_amplification(instruction_set: InstructionSet, state_preparer: InstructionSet, good_space: SubSpace,
                            lower_bound_probability, qargs: [Qubit], ancilla, delta=0.001):
    inverse_of_preparer = state_preparer.get_inverse()
    zero_subspace = SubSpace()
    zero_subspace.add_basis([0 for i in range(len(qargs))])
    # For over 99% success rate:
    num_iterations = math.ceil(math.log2(2 / delta) / math.sqrt(lower_bound_probability)) // 2 + 1
    for i in range(num_iterations):
        partial_reflect_abt_subspace(instruction_set, good_space, get_fp_beta(i + 1, num_iterations, delta),
                                     qargs, ancilla)
        instruction_set.add_instruction_set(inverse_of_preparer)

        partial_reflect_abt_subspace(instruction_set, zero_subspace, -get_fp_alpha(i + 1, num_iterations, delta),
                                     qargs, ancilla)
        instruction_set.add_instruction_set(state_preparer)


def get_fp_alpha(current_iteration, total_iterations, delta):
    gamma = np.cos(np.arccos(1 / delta, dtype=np.complex128) / (total_iterations * 2 + 1), dtype=np.complex128) ** -1

    input_of_cot_inverse = np.tan(2 * math.pi * current_iteration / (total_iterations * 2 + 1), dtype=np.complex128) * np.sqrt(1 - gamma ** 2, dtype=np.complex128)
    input_of_cot_inverse = input_of_cot_inverse.real
    # Returning the cot inverse of the above times 2:
    if input_of_cot_inverse > 0:
        return math.atan(1 / input_of_cot_inverse) * 2
    elif input_of_cot_inverse < 0:
        return (math.atan(1 / input_of_cot_inverse) + math.pi) * 2
    return math.pi


def get_fp_beta(current_iteration, total_iterations, delta):
    return -get_fp_alpha(total_iterations - current_iteration + 1, total_iterations, delta)


def QSP(instruction_set: InstructionSet, qubit: Qubit, processingPhases: [float], signalPhase, signalPhaseMode="a"):
    theta = signalPhase
    if signalPhaseMode == "a":
        theta = -2 * np.arccos(signalPhase)

    for phase in processingPhases[len(processingPhases):0:-1]:
        instruction_set.add_instruction("rz", qubit, [phase])
        instruction_set.add_instruction("rx", qubit, [theta])

    instruction_set.add_instruction("rz", qubit, [processingPhases[0]])


# Assumes projector qubits in 0 state give A where A is the block encoded matrix we're interested in transforming
def QEVT_Encoded(instruction_set: InstructionSet, U: InstructionSet, projectors: [Qubit], phases: [float],
                 ancilla1: Qubit):
    U_inverse = U.get_inverse()
    apply_inverse = False
    for phase in phases[::-1]:
        if not apply_inverse:
            instruction_set.add_instruction_set(U)
        else:
            instruction_set.add_instruction_set(U_inverse)

        apply_inverse = not apply_inverse

        for qubit in projectors:
            instruction_set.add_instruction("x", [qubit], [])

        # Special case when only 1 projector:
        if len(projectors) == 1:
            instruction_set.add_instruction("cx", [projectors[0], ancilla1])
            # Doing 2 rotations by phase instead of a single phase*2 rotation for the sake of parameterization:
            instruction_set.add_instruction("rz", [ancilla1], [phase])
            instruction_set.add_instruction("rz", [ancilla1], [phase])
            instruction_set.add_instruction("cx", [projectors[0], ancilla1])
        else:
            instruction_set.multiControl("x", projectors, ancilla1)
            # Doing 2 rotations by phase instead of a single phase*2 rotation for the sake of parameterization:
            instruction_set.add_instruction("rz", [ancilla1], [phase])
            instruction_set.add_instruction("rz", [ancilla1], [phase])
            instruction_set.multiControl("x", projectors, ancilla1)

        for qubit in projectors:
            instruction_set.add_instruction("x", [qubit], [])


def GetChebyshevBasis(func, degree=12):
    bases = []
    for i in range(degree + 1):

        def integrand(x):
            return func(x) * chebyshevt(i, x) * 1 / sqrt(1 - x ** 2)

        if i == 0:
            bases.append(quad(integrand, -1, 1)[0] / math.pi)
        else:
            bases.append(2 * quad(integrand, -1, 1)[0] / math.pi)

    return bases


# TODO: This is not LCU in its full form
def LCU(Unitary1: InstructionSet, Unitary2: InstructionSet, deciding_qubit: Qubit, name=None):
    if name is None:
        name = "LCU of {} and {}".format(Unitary1.name, Unitary2.name)

    LCU = InstructionSet(name)

    LCU.add_instruction("h", [deciding_qubit])
    LCU.add_instruction_set(Unitary1.get_zero_controlled_version([deciding_qubit]))
    LCU.add_instruction_set(Unitary2.get_controlled_version([deciding_qubit]))
    LCU.add_instruction("h", [deciding_qubit])

    return LCU


## Calculates <psi| H |psi>
## Inputs:
## operator: list of tuples (a_i, U_i) where H = sum_over_i(a_i U_i), U_i s here are InstructionSets
##                                                   Note: can be better represented by a class maybe?
## state: An instruction set U such that
## Registers: Both the operator and the state must have used the same qubits (Registers)
## shots: number of shots to use to get the expectation value
## backend: IBM backend to run on
## p-func: if you want to replace Re(<psi| U |psi>) = (2 * p_zero - 1) with some other function to get the expection of
# some subsystem or for some other mathematical reason the user have worked out (Do not pass this param unless you know what you're doing)
## Note: both the operator and the state must have used the same qubits (Register)******
def get_real_expectation_value(operator, state, state_binding, registers: [QuantumRegister], provider, shots=8192,
                               p_func=None, p_func_args=None, aersim=None):
    if len(operator) == 0:
        raise Exception("Operator can't be an empty list")
    # if len(operator) > 1:
    #     exp_val = 0
    #     for element in operator:
    #         exp_val += get_real_expectation_value([element], state, state_binding, registers, provider, shots=shots, p_func=p_func, p_func_args=p_func_args)
    #     return exp_val

    # TODO: Batu I think here I was trying to create the expectation value circuit for all terms simultaneously so I
    #  can use Qiskit's runtime with batches of circuits to avoid the latency but that wouldn't be necessary if we
    #  use the local simulator. I think a good exercise for you would be to rewrite this function to make sure you
    #  understand how the expectation values are calculate (Hadamard test for each term in the Hamiltonian)
    measurment_qubits = [QuantumRegister(1) for i in range(len(operator))]
    instructions = [InstructionSet() for i in range(len(operator))]

    for i, element in enumerate(operator):
        instructions[i].add_instruction_set(state)
        instructions[i].add_instruction("h", [measurment_qubits[i][0]])
        instructions[i].add_instruction_set(element[1].get_controlled_version([measurment_qubits[i][0]]))
        instructions[i].add_instruction("h", [measurment_qubits[i][0]])

    crs = [ClassicalRegister(1) for i in range(len(operator))]
    circuits = [QuantumCircuit(measurment_qubits[i], crs[i]) for i in range(len(operator))]
    for register in registers:
        for circuit in circuits:
            circuit.add_register(register)

    for i in range(len(operator)):
        instructions[i].load_circuit(circuits[i], state_binding)
        circuits[i].measure(0, 0)

    # runtime_inputs = {
    #     # A circuit or a list of circuits.
    #     # A QuantumCircuit or a list of QuantumCircuits. (required)
    #     'circuits': circuits + [p_func_args],
    #
    #     # Number of repetitions of each circuit, for sampling. . Default: 1024.
    #     'shots': shots,  # int
    #
    #     # Initial position of virtual qubits on physical qubits.
    #     'initial_layout': {},  # dict or list
    #
    #     # Name of layout selection pass ('trivial', 'dense', 'noise_adaptive', 'sabre')
    #     'layout_method': "trivial",
    #
    #     # Name of routing pass ('basic', 'lookahead', 'stochastic', 'sabre').
    #     'routing_method': "basic",
    #
    #     # Name of translation pass ('unroller', 'translator', 'synthesis').
    #     'translation_method': "unroller",
    #
    #     # Sets random seed for the stochastic parts of the transpiler.
    #     'seed_transpiler': 42,  # obviously 😗
    #
    #     # How much optimization to perform on the circuits (0-3). Higher levels
    #     # generate more optimized circuits. Default is 1.
    #     'optimization_level': 1,
    #
    #     # Whether to reset the qubits to the ground state for each shot.
    #     'init_qubits': True,
    #
    #     # Delay between programs in seconds.
    #     'rep_delay': 0,  # float
    #
    #     # Additional compilation options.
    #     'transpiler_options': {},  # dict
    #
    #     # Whether to apply measurement error
    #     # mitigation. Default is False.
    #     'measurement_error_mitigation': False
    # }
    #
    # job = provider.runtime.run(
    #     program_id='circuit-runner',
    #     options={'backend_name': "ibmq_qasm_simulator"},
    #     inputs=runtime_inputs
    # )
    #
    # failed = True
    # while failed:
    #     try:
    #         result = job.result()
    #
    #         failed = False
    #
    #     except:
    #         print("Job Failure")
    #         return get_real_expectation_value(operator, state, state_binding, registers, provider, shots, p_func,
    #                                           p_func_args)

    # counts = [result.get("results")[i].get("data").get("counts") for i in range(len(operator) + 1)]***
    # p_zeros = [counts[i].get("0x0") / shots for i in range(len(operator))]***

    p_zeros = []

    for c in circuits:
        p_zeros.append(q.execute(c, aersim, shots=shots).result().get_counts(0).get('0') / shots)

    ## Assumes shots for p_func_args is the same as shots here
    p_zeros.append(q.execute(p_func_args, aersim, shots=shots).result().get_counts(0).get('0') / shots)

    s = 0
    if p_func is None:
        for i in range(len(operator)):
            s += operator[i][0] * (2 * p_zeros[i] - 1)
        return s

    for i in range(len(operator)):
        # s += operator[i][0] * p_func(p_zeros[i], [counts[-1].get("0x0") / shots])****
        s += operator[i][0] * p_func(p_zeros[i], [p_zeros[-1]])
    return s


# Inputs:
#     U: An InstructionSet representing a unitary U = exp(iH)
#     flag_qubit: The additionally qubit to form a block encoding
#     mode: Determines the type of block encoding
#
# Outputs:
#     A block encoding of cos(H) and sin(H) multiplied by appropriate coefficients using the mode parameter.
#
#     mode = 0:
#
#         [[cos(H), isin(H)],
#          [isin(H), cos(H)]]
#
#     mode = 1:
#
#         [[sin(H), -cos(H)],
#          [cos(H), sin(H)]]
def hermitian_block_encode(U: InstructionSet, flag_qubit: Qubit, mode=0):
    U_dagger = U.get_inverse()

    if mode == 0:
        return LCU(U, U_dagger, flag_qubit, f"Hermitian block encoding of {U.name}, mode = 0")
    elif mode == 1:
        encoding = InstructionSet(name=f"Hermitian block encoding of {U.name}, mode = 1")
        encoding.add_instruction("sdg", [flag_qubit])
        encoding.add_instruction_set(LCU(U, U_dagger, flag_qubit))
        encoding.add_instruction("sdg", [flag_qubit])
        encoding.add_instruction("x", [flag_qubit])
        return encoding


def Unitary_QSVT(instruction_set: InstructionSet, U: InstructionSet, flag: [Qubit], phases: [float],
                 ancilla1: Qubit):
    U_inverse = U.get_inverse()

    block = InstructionSet()
    block.add_instruction_set(U.get_zero_controlled_version([flag]))
    block.add_instruction_set(U_inverse.get_controlled_version([flag]))
    block.add_instruction("h", [flag])
    for phase in phases[::-1]:
        instruction_set.add_instruction_set(block)

        instruction_set.add_instruction("x", [flag], [])

        instruction_set.add_instruction("cx", [flag, ancilla1])
        # Doing 2 rotations by phase instead of a single phase*2 rotation for the sake of parameterization:
        instruction_set.add_instruction("rz", [ancilla1], [phase])
        instruction_set.add_instruction("rz", [ancilla1], [phase])
        instruction_set.add_instruction("cx", [flag, ancilla1])

        instruction_set.add_instruction("x", [flag], [])
