import qiskit as q
from qiskit import IBMQ, QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

from AlgorithmicPrimitives import amplitude_amplification
from InstructionSet import InstructionSet
from SubSpace import SubSpace

# Setting up cloud access
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
# provider.backends() to get all available backends
backend = provider.get_backend('ibmq_qasm_simulator')
# backend = Aer.get_backend('statevector_simulator')
aersim = AerSimulator()

num_qubits = 8
core = QuantumRegister(num_qubits)
ancilla = QuantumRegister(1)

initial = InstructionSet()
for i in range(num_qubits):
    initial.add_instruction("h", [core[i]], [])
    # initial.add_instruction(Instruction("h", 1, 0, []), [core[i]])

good_space = SubSpace()
good_space.add_basis([0 for i in range(num_qubits)])
# good_space.add_basis([1 for i in range(num_qubits)])

instructions = InstructionSet()
instructions.add_instruction_set(initial)
amplitude_amplification(instructions, initial, good_space, 1/(2**num_qubits), core, ancilla[0])
circuit = QuantumCircuit()
circuit.add_register(core)
circuit.add_register(ancilla)
instructions.load_circuit(circuit)

circuit.measure_all()

# executing the circuit on the backend:
job = q.execute(circuit, backend=aersim, shots=8192)
job_monitor(job)  # To monitor the status of the job
result = job.result()

counts = result.get_counts(circuit)

plot_histogram([counts]).savefig("Refactor test")

# const = 3
# for const in range(1, 10):
#     instructions = InstructionSet()
#     for i in range(num_qubits):
#         instructions.add_instruction(Instruction("h", 1, 0, []), [core[i]])
#     for i in range(num_qubits):
#         instructions.add_instruction(Instruction("ry", 1, 0, [math.pi / const]), [core[i]])
#         instructions.add_instruction(Instruction("rx", 1, 0, [math.pi / const]), [core[i]])
#     circuit = QuantumCircuit()
#     circuit.add_register(core)
#     circuit.add_register(ancilla)
#     instructions.load_circuit(circuit)
#     circuit.measure_all()
#     # executing the circuit on the backend:
#     job = q.execute(circuit, backend=backend, shots=8192)
#     job_monitor(job)  # To monitor the status of the job
#     result = job.result()
#
#     counts = result.get_counts(circuit)
#     plot_histogram([counts]).savefig("Experiment " + str(const))

# for i in [2**-j for j in range(25)]:
#     instructions = InstructionSet()
#     instructions.add_instruction_set(initial)
#     amplitude_amplification(instructions, initial, good_space, i, core, ancilla[0])
#     circuit = QuantumCircuit()
#     circuit.add_register(core)
#     circuit.add_register(ancilla)
#     instructions.load_circuit(circuit)
#     circuit.measure_all()
#     # executing the circuit on the backend:
#     job = q.execute(circuit, backend=backend, shots=8192)
#     job_monitor(job)  # To monitor the status of the job
#     result = job.result()
#     counts = result.get_counts(circuit)
#     print(counts["000000"])
#     probabilities[str(i)] = counts["000000"]/8192

# Getting state_vector:
#######################################################################
# backend = Aer.get_backend('statevector_simulator')
# job = q.execute(circuit, backend=backend, shots=1000, memory=True)
# job_result = job.result()
# co = job_result.get_statevector(circuit)[0] \\ [0 is an example]
# print(((co.real)**2 + (co.imag)**2)) \\ print probability
#######################################################################

# circuit.draw(output="mpl", filename="my_circuit2")
# circuit.mcx()
# executing the circuit on the backend:
# job = q.execute(circuit, backend=backend, shots=8192)
# job_monitor(job)  # To monitor the status of the job
# result = job.result()
# counts = result.get_counts(circuit)

# myList = probabilities.items()
# # myList = sorted(myList)
# x, y = zip(*myList)
#
# plt.plot(x, y)
# plt.show()
# # Saving the result as a plot


# plot_histogram([counts]).savefig("Test")
