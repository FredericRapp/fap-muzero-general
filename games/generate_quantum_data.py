import numpy as np

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library.n_local.efficient_su2 import EfficientSU2
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

from games.generate_fashion_mnist import FashionMnist


class IsingEncoding:
    def __init__(self, num_qubits, num_su2_layers=None, su2_random_seed=42):
        self.num_qubits = num_qubits

        if num_su2_layers is not None:
            self.num_su2_layers = num_su2_layers
        else:
            if self.num_qubits in range(2, 13):
                # In the original paper, the number of layers is chosen such
                # that the number of parameters is ~90 at all times
                num_layers_list = [15, 10, 7, 6, 5, 4, 4, 3, 3, 3, 3]
                self.num_su2_layers = num_layers_list[self.num_qubits - 2]
            else:
                print('Please set num_su2_layers manually')

        np.random.seed(su2_random_seed)
        self.su2_parameters = np.random.uniform(
            0, 2 * np.pi, self.num_su2_layers * self.num_qubits * 3)

        self.circuit = QuantumCircuit(self.num_qubits)
        self.x_parameter_vector = ParameterVector('x', self.num_qubits)

    def create_encoding_circuit(self):
        if self.num_su2_layers != 0:
            variational_layer = EfficientSU2(
                num_qubits=self.num_qubits, su2_gates=['rx','ry', 'rz'], reps=self.num_su2_layers-1)
        self.add_hadamard_wall()
        self.add_uz()
        self.add_hadamard_wall()
        self.add_uz()
        if self.num_su2_layers != 0:
            self.circuit.append(variational_layer, range(self.num_qubits))

    def add_uz(self):
        self.create_z_layer()
        self.create_zz_layer()

    def add_hadamard_wall(self):
        for i in range(self.num_qubits):
            self.circuit.h(i)

    def create_z_layer(self):
        for i in range(self.num_qubits):
            self.circuit.rz(self.x_parameter_vector[i], i)

    def create_zz_layer(self):
        for i in range(self.num_qubits):
            j = i + 1
            while (j > i) and j < self.num_qubits:
                self.circuit.rzz(
                    self.x_parameter_vector[i] * self.x_parameter_vector[j], i, j)
                j += 1

    def assign_x_values(self, x_values):
        parameter_dict = {}
        for i in range(self.num_qubits):
            parameter_dict[self.x_parameter_vector[i]] = x_values[i]
        return self.circuit.bind_parameters(parameter_dict)

    def calculate_output(self, data):
        if len(data.shape) == 2:
            num_data_points = data.shape[0]
        else:
            num_data_points = 1

        estimator = Estimator()
        observable = SparsePauliOp('Z' + (self.num_qubits - 1) * 'I')

        result_list = []
        for i in range(num_data_points):
            circuit_with_bound_x = self.assign_x_values(data[i, :])
            job = estimator.run(
                circuit_with_bound_x, observable, parameter_values=self.su2_parameters)
            result = job.result()
            result_list.append(result.values[0])
        return result_list

def create_quantum_fmnist(
        num_training,
        num_test, num_components=5,
        category_filter_list=None,
        num_su2_layers=None,
        su2_random_seed=42):

    fashion_mnist = FashionMnist()
    X_train, y_train, X_test, y_test = fashion_mnist.load_data()
    if category_filter_list is not None:
        X_train, y_train = fashion_mnist.filter_by_label(X_train, y_train, category_filter_list)
        X_test, y_test = fashion_mnist.filter_by_label(X_test, y_test, category_filter_list)

    
    X_train_reduced, X_test_reduced = fashion_mnist.preprocess_train_and_test_data(
        X_train, X_test, n_components=num_components)

    ising_encoding = IsingEncoding(
        num_qubits=num_components, num_su2_layers=num_su2_layers, su2_random_seed=su2_random_seed)
    ising_encoding.create_encoding_circuit()

    X_train_encoded = X_train_reduced[:num_training, :]
    X_test_encoded = X_test_reduced[:num_test, :]
    y_train_encoded = ising_encoding.calculate_output(X_train_encoded)
    y_test_encoded = ising_encoding.calculate_output(X_test_encoded)
    return X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded
