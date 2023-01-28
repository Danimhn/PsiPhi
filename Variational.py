import time
import random
from qiskit.providers.aer import AerSimulator, Aer
import numpy
import scipy.optimize as spo
from AlgorithmicPrimitives import *
from qiskit import IBMQ, QuantumRegister, QuantumCircuit

# Setting up cloud access
IBMQ.load_account()
provider = IBMQ.get_provider()
aersim = AerSimulator()

num_main_qubits = 12
n_ansatz = 5

backend = provider.get_backend('ibmq_qasm_simulator')

best = math.inf
seed = 34
np.random.seed(seed)
random.seed(seed)


## TODO: Batu this returns an actual matrix but for this project we'll need to represent the Hamiltonian as a linear combination of unitaries (Pauli tensor products)
def create_random_HB(num_qubits, num_edges):
    Z = numpy.array([[1, 0], [0, -1]])
    I = numpy.array([[1, 0], [0, 1]])
    X = numpy.array([[0, 1], [1, 0]])
    Y = numpy.array([[0, -1j], [1j, 0]])
    paulis = [I, X, Y, Z]
    base = [I for i in range(num_qubits)]
    matrix = numpy.zeros((2 ** num_qubits, 2 ** num_qubits), dtype='complex128')

    for k in range(num_edges):
        pi = random.randint(0, 3)
        ps = random.randint(0, 3)
        i = random.randint(0, num_qubits - 1)
        s = random.randint(0, num_qubits - 1)

        base[i] = paulis[pi]
        base[s] = paulis[ps]
        coef = random.gauss(0, 1)
        m = base[0]

        for term in base[1:]:
            m = numpy.kron(term, m)

        matrix += coef * m

        base[i] = I
        base[s] = I

    return matrix


# Objective function:
def f(variables, args):
    U = args[0]
    num_params = args[1]
    qargs = args[2]
    matrix = args[3]
    ancillas = args[4]

    binding = {}
    for i in range(num_params):
        binding.update({f'{i}': variables[i]})

    circuit = QuantumCircuit()
    circuit.add_register(qargs)
    circuit.add_register(ancillas)

    U.load_circuit(circuit, parameter_binding_dict=binding)

    # # Timing:
    # tic = time.perf_counter()
    # ...
    # toc = time.perf_counter()
    # print(f'Loading circuit took {toc - tic}')

    # TODO: Batu here I'm using the exact vector representation to get the expected value, but for this project we
    #  want to estimate the expected value through measurements. Take a look at get_real_expectation_value function
    #  in AlgorithmicPrimitives.py
    backend = Aer.get_backend('statevector_simulator')

    job = q.execute(circuit, backend=backend, shots=1000, memory=True, optimization_level=0, seed_simulator=seed)

    job_result = job.result()
    co = job_result.get_statevector(circuit)  # [0 is an example]

    global num_main_qubits
    norm = np.linalg.norm(co[:2 ** num_main_qubits])
    psi = co[:2 ** num_main_qubits] / norm
    expected = np.matmul(psi.conj().T, np.matmul(matrix, psi))

    return expected.real, psi, norm


def find_good_initial_point(num_initials, num_params, objective_params, shots=8192):
    vals = []
    for i in range(num_initials):
        if i % 100 == 0:
            print(f"i = {i}")
        p = numpy.random.rand(num_params) * math.pi
        vals.append((p, f(p, objective_params)[0]))

    vals = sorted(vals, key=lambda x: x[1])
    initiatal_simp = numpy.zeros((num_params + 1, num_params))
    for i in range(num_params + 1):
        initiatal_simp[i] = vals[i][0]

    return [vals[i][1] for i in range(num_params + 1)], initiatal_simp


def get_ansatz(num_qubits, num_ansatz):
    layer_per_ansatz = 3
    params_per_ansatz = num_qubits * layer_per_ansatz
    ansatzes = []
    for k in range(num_ansatz):
        ansatz = InstructionSet()
        offset = params_per_ansatz * k
        for i in range(num_qubits):
            ansatz.add_instruction("ry", [main_qubits[i]],
                                   params=[str(offset + i)])
            ansatz.add_instruction("rz", [main_qubits[i]], params=[str(offset + num_main_qubits + i)])

            if i != num_main_qubits - 1:
                ansatz.multiControl("rx", [main_qubits[i]], main_qubits[i + 1],
                                    params=[str(offset + 2 * num_main_qubits + i)])
            else:
                ansatz.multiControl("rx", [main_qubits[i]], main_qubits[0],
                                    params=[str(offset + 2 * num_main_qubits + i)])

        ansatzes.append(ansatz)

    return params_per_ansatz * num_ansatz, ansatzes


## TODO: Batu, won't be needing this
def load_unitary_QEVT_ansatz(instruction_set, ansatzes, d, num_ansatzes_params, flag_qubit):
    num_QEVT_layers = len(ansatzes) // 2

    for j, ans in enumerate(ansatzes):
        if j % 2 == 0:
            instruction_set.add_instruction_set(ans)
        else:
            k = j // 2  # This is the (k + 1)th  QEVT layer
            Unitary_QSVT(instruction_set, ans, flag_qubit, [str(num_ansatzes_params + i + k * d) for i in range(d)],
                         ancilla)
            instruction_set.add_instruction("ry", [flag_qubit],
                                            params=[str(num_ansatzes_params + num_QEVT_layers * d + k)])

    return num_ansatzes_params + num_QEVT_layers * (d + 1)


# Test create_random_FC_TIM returns Hermitian:
main_qubits = QuantumRegister(num_main_qubits)
ancillas = QuantumRegister(2)  # First is flag second is ancilla
matrix = create_random_HB(num_main_qubits, 500)
# assert numpy.allclose(matrix, numpy.asmatrix(matrix).getH())
print("Hamiltonian:")
print(matrix)
print("5 smallest eigenvalues and smallest eigenvector:")
print(numpy.linalg.eigh(matrix)[:5][0])

n, ansatzes = get_ansatz(num_main_qubits, n_ansatz)


class Wrapper(object):
    def __init__(self, f, file):
        self.f = f
        self.last_x = None
        self.last_f = None
        self.ncall = 0
        self.best_x = None
        self.best_psi = None
        self.best = math.inf
        self.file = file
        self.avgnorm = 0

    def __call__(self, x, args):
        self.last_x = x
        self.last_f, psi, norm_ = self.f(x, args)
        self.avgnorm += norm_
        self.ncall += 1
        if self.last_f < self.best:
            self.best = self.last_f
            self.best_x = x
            self.best_psi = psi
            # print(x)
        if self.ncall % 100 == 0:
            print(f"{self.ncall}. best: {self.best} avg_norm:{self.avgnorm / 100}")
            if self.file is not None:
                self.file.write(f"{self.ncall}. best: {self.best} avg_norm:{self.avgnorm / 100}\n")
            self.avgnorm = 0
        return self.last_f


flag_qubit = ancillas[0]
ancilla = ancillas[1]

# block_encoded = hermitian_block_encode(ansatzes[], flag_qubit, mode=0) block_encoded2 = hermitian_block_encode(
# ansatz_4, flag_qubit, mode=0)  # Can experiment with different modes.
# TODO: Batu this is the training I had setup
#  to collect certain information I needed, modifiy it to collect the data you'll need.
for i in range(-1, 4):
    d = i  # degree of poly
    best = math.inf
    if (i % 2 == 1 and i > 2) or i == -1:
        print(f'd={d}')
        file = open(f"n=14d={d}_NelderMead30000fval_1500init_optionalEntangle_layersmultiplier=2.txt", "w")
        func = Wrapper(f, file)
        instruction_set = InstructionSet()

        if d > 0:
            num_params = load_unitary_QEVT_ansatz(instruction_set, ansatzes, d, n, flag_qubit)
            v, initital_simp = find_good_initial_point(1500, num_params,
                                                       [instruction_set, num_params, main_qubits, matrix, ancillas])

            result = spo.minimize(func, x0=initital_simp[0],
                                  args=[instruction_set, num_params, main_qubits, matrix, ancillas], method='BFGS',
                                  options={'gtol': 1e-05, 'eps': 1e-07, 'maxiter': 500, 'disp': True,
                                           'return_all': True})
        else:
            for ans in ansatzes:
                instruction_set.add_instruction_set(ans)

            v, initital_simp = find_good_initial_point(1500, n, [instruction_set, n, main_qubits, matrix, ancillas])
            result = spo.minimize(func, x0=initital_simp[0], args=[instruction_set, n, main_qubits, matrix, ancillas],
                                  method='BFGS',
                                  options={'gtol': 1e-05, 'eps': 1e-07, 'maxiter': 500, 'disp': True,
                                           'return_all': True})

        print(f'd = {d} y = {result.fun} number of iterations = {result.nit} function evals = {result.nfev}')
        file.close()
