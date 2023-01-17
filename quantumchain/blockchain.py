"""Blockchain module"""

from os import makedirs
from os.path import join, exists
from shutil import copy2, copy, rmtree

from pandas import DataFrame, concat
from numpy import array, ndarray, ones, zeros, unravel_index, argmax
from numpy.random import random_sample

from qiskit import Aer, QuantumCircuit, execute, assemble
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.extensions import Initialize
from qiskit.quantum_info import state_fidelity
from qiskit.tools.monitor import job_monitor  # noqa

from .node import Node


class Blockchain:
    """
    Class of a full blockchain. When created it sets up the
    initial folders and creates the node objects.

    Attributes
    ----------

    node_number : int
        number of nodes in the blockchain (integer)
    nodes : list[int]
        List of node indices
    node_list : list[Node]
        A list of node objects, one for each node in the blockchain.
    tran_per_block : int
        The number of transactions per block.
    rand_length : int
        The length of the random numbers the blockchain generates.

    """

    def __init__(self, node_number: int, tran_per_block: int, rand_length: int):
        """
        Parameters
        ----------

        node_number : int
            The number of nodes in the blockchain.
        tran_per_block : int
            The number of transactions per block.
        rand_length : int
            The length of the random numbers the blockchain generates.
        """

        self.node_number = node_number
        self.nodes = list(range(node_number))
        self.tran_per_block = tran_per_block
        self.rand_length = rand_length

        # Create initialize transactions and first block csv
        self._initialize_files()

        # Create one node object for each node
        self.node_list = []
        for i in range(self.node_number):
            self.node_list.append(Node(self.node_number, i, self.tran_per_block))

            # Copy inital folders and files for each node

            node_folder = join("Data", f"Node{i}")
            block_folder = join(node_folder, "Blocks")

            if exists(node_folder):
                rmtree(node_folder)

            makedirs(node_folder, exist_ok=True)
            makedirs(block_folder, exist_ok=True)

            copy2(join("Data", "Initialize", "transactions.csv"), node_folder)
            copy2(join("Data", "Initialize", "fidelity.csv"), node_folder)
            copy2(join("Data", "Initialize", "block1.csv"), block_folder)

    def _initialize_files(self):
        """Initialize transaction file with 20 coins to every node in the first
        transaction and zero in the rest transactions of the first block.
        Iniitialize fidelities to random numbers between 0 and 1."""

        # Create data as pandas dataframe
        initial_tran = DataFrame(
            {
                "Node": list(range(self.node_number)),
                "Transaction 1": list(ones(self.node_number) * 20),
            },
            dtype=object,
        )
        initial_tran.set_index("Node", inplace=True)

        initial_fid = DataFrame({"Node": list(range(self.node_number))}, dtype=object)
        initial_fid.set_index("Node", inplace=True)

        for i in range(2, self.tran_per_block + 1):
            initial_tran[f"Transaction {i}"] = list(zeros(self.node_number))
        for i in self.nodes:
            initial_fid[f"prev_fidelity {i}"] = list(random_sample(self.node_number))

        # Create Initialize folder and copy the csv's
        makedirs("Initialize", exist_ok=True)
        initial_tran.to_csv(join("Data", "Initialize", "transactions.csv"))
        initial_fid.to_csv(join("Data", "Initialize", "fidelity.csv"))

        # Initialize first block file for every node with 0 coins in old wallet
        # and 20 coins in wallet after the first transaction
        initial_block = DataFrame(
            {
                "Node": list(range(self.node_number)),
                "Old Wallet": list(zeros(self.node_number)),
                "Wallet": list(ones(self.node_number) * 20),
                "Payment": list(zeros(self.node_number)),
                "Random number": ["00000" for i in range(self.node_number)],
            },
            dtype=object,
        )
        initial_block = concat([initial_tran, initial_block, initial_fid], axis=1)
        initial_block.to_csv(join("Data", "Initialize", "block1.csv"))

    def random_numbers(self):
        """Create one random number sequence for each node. One can obtain one sequence from
        the rest, which can be used to verify that the sequences were obtained by the prescribed
        precedure in this function.

        References
        ----------
        [1] Janusz E., Jacak, Witold A. Jacak, Wojciech A. Donderowicz & Lucjan Jacak, (2020).
        "Quantum random number generators with entanglement for public randomness testing". Nature.
        """

        def random_digits():
            """Create the individual numbers of the sequences of random numbers using a
            quantum circuit with entangled states.
            """
            circ = QuantumCircuit(self.node_number, self.node_number)

            for i in range(self.node_number - 1):
                circ.h(i)
                circ.cx(i, self.node_number - 1)
                circ.measure(i, i)
            circ.measure(self.node_number - 1, self.node_number - 1)

            # backend = provider.get_backend('ibmq_belem')
            backend = Aer.get_backend("qasm_simulator")
            job = execute(circ, backend, shots=1)
            # job_monitor(job)
            return job.result().get_counts()

        # Initialize and create the list of strings of random sequences
        rand_string = [""] * self.node_number
        for _ in range(self.rand_length):
            # Measure the quantum circuit and get the resulting bits
            measure = list(random_digits().keys())[0]
            # Join each the characters to create strings
            for i in range(self.node_number):
                rand_string[i] += measure[i]
        return rand_string

    @staticmethod
    def send_states(angles: list[float, float]) -> list[float, float]:
        """Each nodes sends its block state (obtained through hashing) to the rest of nodes by a
        Quantum Teleportation protocol, which ensures the blockchain is made up of Quantum Computers
        and Quantum Information channels. The function returns the qubit state after teleportation
        (which should be the same if everything went OK)."""

        def create_bell_pair(circ, q1, q2):
            """Creates a bell pair in quantum circuit using qubits a and b"""
            circ.h(q1)
            circ.cx(q1, q2)

        def node1_gates(circ, psi, q1):
            """Create sending node gates"""
            circ.cx(psi, q1)
            circ.h(psi)

        def measure_and_send(circ, q1, q2):
            """Measures qubits q1 and q2 and 'sends' the results to node2"""
            circ.barrier()
            circ.measure(q1, 0)
            circ.measure(q2, 1)

        def node2_gates(circ, qubit, crz, crx):
            """Create receiving node gates"""
            circ.x(qubit).c_if(crx, 1)
            circ.z(qubit).c_if(crz, 1)

        # Generate the qubit we want to send with the parameters θ and ϕ
        state_circ = QuantumCircuit(1)
        state_circ.u(angles[0], angles[1], 0, 0)

        # Measure the circuit after we rotated |0> vector
        backend = Aer.get_backend("qasm_simulator")
        state_circ.save_statevector()
        qobj = assemble(state_circ)  # Create a Qobj from the circuit for the simulator to run
        result = backend.run(qobj).result()
        psi = result.get_statevector()

        # Once we have our state ready to send we begin the teleportation protocol
        init_gate = Initialize(psi)
        init_gate.label = "init"

        # Protocol uses 3 qubits and 2 classical bits in 2 different registers
        qr = QuantumRegister(3, name="q")
        crz, crx = ClassicalRegister(1, name="crz"), ClassicalRegister(1, name="crx")
        teleport_circ = QuantumCircuit(qr, crz, crx)

        # Step 1
        teleport_circ.append(init_gate, [0])
        create_bell_pair(teleport_circ, 1, 2)
        # Step 2
        node1_gates(teleport_circ, 0, 1)
        # Step 3
        measure_and_send(teleport_circ, 0, 1)
        # Step 4
        node2_gates(teleport_circ, 2, crz, crx)

        # Finally we determine the received state
        teleport_circ.save_statevector()  # Tell simulator to save statevector
        qobj = assemble(teleport_circ)  # Create a Qobj from the circuit for the simulator to run
        result = backend.run(qobj).result()
        out_state = result.get_statevector()

        # We obtain final state in two parts
        part1 = array(out_state)[0:4]
        part2 = array(out_state)[4:8]

        return [max(part1.min(), part1.max(), key=abs), max(part2.min(), part2.max(), key=abs)]

    def fidelity_table(self, states: ndarray) -> ndarray:
        """Given the states of each node's last block, obtain the fidelities.
        Each node compares its vector to other nodes vector. One's own fidelity
        is set to 0 by default. Otherwise, it would always be 1.

        Input:
        - states: array of each node state
        Output:
        - fid_table: table of fidelities as a numpy array
        """
        fid_table = zeros((self.node_number, self.node_number))
        for i in range(self.node_number):
            fid_table[i, i] = 0
            for j in range(i):
                fid = state_fidelity(states[i], states[j], validate=True)
                fid_table[i, j] = fid
                fid_table[j, i] = fid

        # We write the fidelity table that each node finds in its fidelity csv
        for node in self.node_list:
            node.write_fidelity(fid_table)

        return array(fid_table)

    @staticmethod
    def get_winners(fid_table: ndarray) -> list:
        """Check fidelities table and select the winners."""

        fid_table = array(fid_table)
        max_idx = list(unravel_index(argmax(fid_table, axis=None), fid_table.shape))

        return max_idx

    def _update_blocks(self, winners: list) -> None:
        """Tell nodes the winners and overwrite nodes last block with first winner's block.
        Both winners will receive a monetary prize."""

        winner_path = join(
            "Data", f"Node{winners[0]}", "Blocks", f"block{self.node_list[0].block_count}.csv"
        )

        for i, node in enumerate(self.node_list):
            node.update_winners(winners)
            if i == winners[0]:
                continue
            copy(winner_path, join("Data", f"Node{i}", "Blocks"))

    def _solve_block(self):
        """Once a block is solved, begin the process of solving
        the block by each node and determining the winners."""

        # get the random numbers generated by each node
        rand_num = []
        for i in self.nodes:
            rand_num.append(self.random_numbers())
        rand_num = array(rand_num, dtype=object)

        # Each node sends one random number to another node (including itself)
        # That is, each node receives a random number from each node (including itself)
        # Then each node creates its block
        for i, node in enumerate(self.node_list):
            node.create_block(list(rand_num[i, :]))

        # Each node sends its block's quantum hashed state to the rest of the nodes
        states = []
        for node in self.node_list:
            states.append(Blockchain.send_states(node.state_parameters(node.block_path)))

        # Each node calculates fidelities and determine the winners
        fid_table = self.fidelity_table(states)
        winners = self.get_winners(fid_table)

        # Tell each block who whon and update winners
        self._update_blocks(winners)  # update all nodes blocks with one of the winners block

    def add_transactions(self, node_send, send, node_receive):
        """Add transactions to the blockchain.

        Input:
        - node_send: list of nodes sending coins
        - send: list of coins sent (one corresponding to each node)
        - node_receive: list of nodes that receive the coins
        """

        for i in range(len(send)):

            for node in self.node_list:
                node.transaction(node_send[i], send[i], node_receive[i])

            # All the nodes receive the same transactions. If a block is to be filled then
            # block resolving process begins.

            if (self.node_list[self.node_number - 1].tran_count % self.tran_per_block) == 0:
                self._solve_block()
