from qiskit import QuantumRegister
from qiskit.circuit import Instruction, QuantumCircuit, Qubit, Clbit


# Params that are not of type _Param or have a numerical value, only come from users!
class InstructionSet:
    class _Param:
        def __init__(self, identifier, coef=1):
            self.identifier = identifier
            self.coef = coef

        def get_identifier(self):
            return self.identifier

        def get_coef(self):
            return self.coef

        def set_coef(self, val):
            self.coef = val

    def __init__(self, name=""):
        self.name = name
        self.instructions = []
        self.multiControlAncilla = None
        self.parameters = []

    def get_name(self):
        return self.name

    def get_parameters(self):
        return self.parameters

    # Assumes no cryy, cxx and so on
    def add_instruction(self, name, qargs: [Qubit], params=None, cargs: [Clbit] = None):
        if params is None:
            params = []
        else:
            # Assumption: params that are not numerical and are not type _Param only come from users!
            # Checking for non numerical params that can be passed in later
            for i, param in enumerate(params):
                if isinstance(param, self._Param):
                    if param.get_identifier() not in self.parameters:
                        self.parameters.append(param.get_identifier())
                elif not isinstance(param, (int, float)):
                    if param not in self.parameters:
                        self.parameters.append(param)
                    params[i] = self._Param(param)
        count = 0
        while name[count] == 'c':
            count += 1
        if count == 0:
            self.instructions.append([name, qargs, params, cargs])
        else:
            self.multiControl(name[count:], qargs[:-1], qargs[-1], params)

    # Doesn't yet support U
    def get_inverse(self):
        inverse = InstructionSet()
        for operation in reversed(self.instructions):
            if type(operation[0]) == str and operation[0][:2] == "mc":
                if operation[0][2:] in ['x', 'y', 'z', 'h', 'swap']:
                    inverse.multiControl(operation[0][2:], operation[1], operation[2], operation[3])
                elif operation[0][2:] in ['s', 't']:
                    inverse.multiControl(operation[0][2:] + 'dg', operation[1], operation[2], operation[3])
                elif operation[0][2:] in ['sdg', 'tdg']:
                    inverse.multiControl(operation[0][2:-2], operation[1], operation[2], operation[3])
                elif operation[0][2:] in ['rx', 'ry', 'rz']:
                    new_param = self._Param(operation[3][0].get_identifier(), coef=-1 * operation[3][0].get_coef())
                    inverse.multiControl(operation[0][2:], operation[1], operation[2], [new_param])
            elif operation[0] in ['x', 'y', 'z', 'h', 'swap']:
                inverse.add_instruction(operation[0], operation[1], operation[2], operation[3])
            elif operation[0] in ['s', 't']:
                inverse.add_instruction(operation[0] + 'dg', operation[1], operation[2], operation[3])
            elif operation[0] in ['sdg', 'tdg']:
                inverse.add_instruction(operation[0][:-2], operation[1], operation[2], operation[3])
            elif operation[0] in ['rx', 'ry', 'rz']:
                new_param = self._Param(operation[2][0].get_identifier(), coef=-1 * operation[2][0].get_coef())
                inverse.add_instruction(operation[0], operation[1], [new_param], operation[3])

        return inverse

    def get_instructions(self):
        return self.instructions

    def add_instruction_set(self, instruction_set):
        for operation in instruction_set.get_instructions():
            self.instructions.append(operation)
        for param in instruction_set.get_parameters():
            if param not in self.parameters:
                self.parameters.append(param)

    def multiControl(self, gate, control_qubits, target_qubit, params=None):
        if params is None:
            params = []
        else:
            # Assumption: params that are not numerical and are not type _Param only come from users!
            # Checking for non numerical params that can be passed in later
            for i, param in enumerate(params):
                if isinstance(param, self._Param):
                    if param.get_identifier() not in self.parameters:
                        self.parameters.append(param.get_identifier())
                elif not isinstance(param, (int, float)):
                    if param not in self.parameters:
                        self.parameters.append(param)
                    params[i] = self._Param(param)
        self.instructions.append(["mc" + gate, control_qubits, target_qubit, params])

    def mcx(self, control_qubits, target_qubit, ancilla):
        self.instructions.append(["mcx", control_qubits, target_qubit, ancilla, []])

    def load_circuit(self, circuit: QuantumCircuit, parameter_binding_dict: dict = None):
        if parameter_binding_dict is None and len(self.parameters) != 0:
            raise Exception("Instruction set is parametrized but no binding dictionary provided")

        ancilla_register_added = False
        for operation in self.get_instructions():
            binded_params = []
            if type(operation[0]) == str and operation[0][:2] == "mc":
                if operation[0] == "mcx":
                    circuit.mcx(operation[1], operation[2])
                else:
                    # Binding parameters with the assumption that all are either _Param or numerical at this point:
                    for param in operation[3]:
                        if isinstance(param, self._Param):
                            if parameter_binding_dict.get(param.get_identifier()) is None:
                                raise Exception("No binding provided for parameter '" + str(param.get_identifier())
                                                + "'")
                            else:
                                p = parameter_binding_dict.get(param.get_identifier()) * param.get_coef()
                                binded_params.append(p)
                        else:  # Param is numerical
                            assert isinstance(param, (int, float))
                            binded_params.append(param)
                    if len(operation[1]) > 1:
                        if not ancilla_register_added:
                            self.multiControlAncilla = QuantumRegister(1)
                            circuit.add_register(self.multiControlAncilla)
                            ancilla_register_added = True
                        self.ncontrolled_operation(circuit, operation[1], operation[2], operation[0][2:],
                                                   self.multiControlAncilla[0], binded_params)
                    else:
                        self.ncontrolled_operation(circuit, operation[1], operation[2], operation[0][2:],
                                                   None, binded_params)
            else:
                # Binding parameters with the assumption that all are either _Param or numerical at this point:
                for param in operation[2]:
                    if isinstance(param, self._Param):
                        if parameter_binding_dict.get(param.get_identifier()) is None:
                            print(type(param.get_identifier()))
                            raise Exception("No binding provided for parameter '" + str(param.get_identifier())
                                            + "'")
                        else:
                            p = parameter_binding_dict.get(param.get_identifier()) * param.get_coef()
                            binded_params.append(p)
                    else:  # Param is numerical
                        assert isinstance(param, (int, float))
                        binded_params.append(param)
                numcl = 0 if (operation[3] is None) else len(operation[3])
                circuit.append(Instruction(operation[0], len(operation[1]), numcl, binded_params), operation[1],
                               operation[3])

        self.multiControlAncilla = None

    def get_controlled_version(self, control_qubits: [Qubit]):
        if self.get_name() != "Default":
            controlled = InstructionSet("Controlled " + self.get_name())
        else:
            controlled = InstructionSet()

        for operation in self.get_instructions():
            if type(operation[0]) == str and operation[0][:2] == "mc":
                controlled.multiControl(operation[0][2:], operation[1] + control_qubits, operation[2], operation[3])
            else:
                controlled.multiControl(operation[0], control_qubits, operation[1][0], operation[2])

        return controlled

    def get_zero_controlled_version(self, control_qubits: [Qubit]):
        if self.get_name() != "Default":
            controlled = InstructionSet("Zero Controlled " + self.get_name())
        else:
            controlled = InstructionSet()

        for control_qubit in control_qubits:
            controlled.add_instruction("x", [control_qubit])

        controlled.add_instruction_set(self.get_controlled_version(control_qubits))

        for control_qubit in control_qubits:
            controlled.add_instruction("x", [control_qubit])

        return controlled

    def ncontrolled_operation(self, circuit: QuantumCircuit, control_qubits, target_qubit, gate, ancilla,
                              params=None):
        if len(control_qubits) == 0:
            raise Exception("Length of control qubits can't be 0")
        elif len(control_qubits) == 1:
            if params is not None:
                circuit.append(Instruction("c" + gate, 2, 0, params), [control_qubits[0], target_qubit])
            else:
                circuit.append(Instruction("c" + gate, 2, 0, []), [control_qubits[0], target_qubit])
        else:
            # Doing an mcx on the ancilla qubit using the target qubit as an ancilla, and then using the ancilla(that was
            # the target in the mcx) as the control qubit to perform the desired control operation on the target:
            # TODO:
            # circuit.mcx(control_qubits, ancilla, [target_qubit], mode="recursion")
            circuit.mcx(control_qubits, ancilla)
            if params is not None:
                circuit.append(Instruction("c" + gate, 2, 0, params), [ancilla, target_qubit])
            else:
                circuit.append(Instruction("c" + gate, 2, 0, []), [ancilla, target_qubit])
            # TODO:
            # circuit.mcx(control_qubits, ancilla, [target_qubit], mode="recursion")
            circuit.mcx(control_qubits, ancilla)

# TODO: add the ability to add group instructions like applying H to the first 10 qubits or something
# TODO: add_controlled_instruction_set currently can't handle rxx, rzz swap and other double qubit operations
# TODO: Allow parameterized qubits as well! Helps with not having to pass around registers?
# TODO: Make fields private
