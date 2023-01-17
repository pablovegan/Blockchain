"""Node module."""

from os.path import join
from hashlib import sha256

from numpy import ndarray, zeros, ones, array, round, pi
from pandas import DataFrame, read_csv, concat


class Node:
    """
    Class of a node in a blockchain. Creating a blockchain object of n nodes
    will create n node objects.

    Attributes
    ----------
    nodes : int
        The number of nodes in the blockchain.
    node_idx : int
        The number associated with current node.
    block_path : srt
        The path of the current block.
    tran_path: path of the transactions csv
    fid_path: path of the fidelity csv
    tran_per_block: fixed number of transactions per block
    tran_count: count of the number of transactions made
    winners: winners of the last block

    Methods
    -------
    transaction: create a transaction between nodes
    create_block: creates a new block with the stored transactions and previous block fidelities.
    update_winners: update winners after block is solved
    write_fidelity: rewrites current fidelity in a file
    sha256_block: pass the block through sha256 hash function
    state_parameters: create block quantum state parameters from hash
    """

    def __init__(self, nodes: int, node_idx: int, tran_per_block: int):
        """Initialize block class. Initializes block number to 1, sets first block's path,
        number of nodes in the blockchain and transaction count.

        Parameters
        ----------
        nodes: number of nodes in the blockchain
        node_idx: current node index (integer)
        tran_per_block: transactions per block
        """
        self.nodes = list(range(nodes))
        self.node_idx = node_idx
        self.block_path = join("Data", f"Node{node_idx}", "Blocks", "block1.csv")
        self.tran_path = join("Data", f"Node{node_idx}", "transactions.csv")
        self.fid_path = join("Data", f"Node{node_idx}", "fidelity.csv")
        self.tran_per_block = tran_per_block
        self.block_count = 1
        self.tran_count = tran_per_block
        self.winners = [0, 1]

    def transaction(self, node_1: int, sends: float, node_2: int):
        """
        Add a transaction from node_1 to node_2 sending an amount of coins.

        Parameters
        ----------
        node_1: integer corresponding to the sending node's number
        sends: coins node_1 sends to node_2
        node_2: integer corresponding to the receiving node's number
        """
        if node_1 == node_2:
            raise Exception("Error, node cannot send coins to itself.")

        prev_block = read_csv(self.block_path)
        prev_block.set_index("Node", inplace=True)

        if prev_block.loc[node_1, "Wallet"] < sends:
            raise Exception("Error, not enough funds in wallet.")

        # Authorised transaction
        self.tran_count += 1

        list_tran = zeros(len(self.nodes))
        list_tran[node_1] = -float(sends)
        list_tran[node_2] = float(sends)

        # Save transactions to the transaction csv
        df_tran = read_csv(self.tran_path)
        df_tran.set_index("Node", inplace=True)

        tran_header = f"Transaction {self.tran_count}"
        df_tran[tran_header] = list_tran
        df_tran.to_csv(self.tran_path)

    def create_block(self, random_numbers: list[str]):
        """
        Write block information in a text file: block number, previous state and two transactions.

        Parameters
        ----------
        random_numbers: strings list
        """
        # Read necessary csv files: transactions.csv, fidelity.csv, block.csv
        prev_block = read_csv(self.block_path)
        prev_block.set_index("Node", inplace=True)

        prev_fid = read_csv(self.fid_path)
        prev_fid.set_index("Node", inplace=True)

        new_block = DataFrame(
            {
                "Node": self.nodes,
                "Old Wallet": prev_block["Wallet"].tolist(),
                "Random number": random_numbers,
            },
            dtype=object,
        )
        new_block.set_index("Node", inplace=True)

        # Add transactions to the block
        df_tran = read_csv(self.tran_path)

        coins_update = zeros(len(self.nodes))
        for i in range(self.tran_count - self.tran_per_block + 1, self.tran_count + 1):
            tran_header = f"Transaction {i}"
            tran_i = df_tran[tran_header].tolist()
            new_block[tran_header] = tran_i
            coins_update += array(tran_i)

        # Nodes get payment for solving previous block
        # fid = new_block['prev_fidelity'].tolist()
        # max_fidelity = fid.index(max(fid))

        payment = 0.2 * ones(len(self.nodes))
        payment[self.winners[0]] = 0.6
        payment[self.winners[1]] = 0.6

        new_block["Payment"] = list(payment)
        old_wallet = array(new_block["Old Wallet"].tolist())
        new_block["Wallet"] = list(round(old_wallet + coins_update + payment, 4))

        # Add previous block fidelities to the block
        new_block = concat([new_block, prev_fid], axis=1)

        # Update path and save block
        self.block_path = join("Data", f"Node{self.node_idx}", "Blocks", f"block{self.block_count + 1}.csv")
        new_block.to_csv(self.block_path)
        self.block_count += 1

    def update_winners(self, winners: list[int, int]):
        """Update winners attribute in node.

        Parameters
        ----------
        winners: list of 2 winners
        """
        self.winners = winners

    def write_fidelity(self, fid_table: ndarray):
        """Write fidelity table into fidelity.csv"""

        fid = DataFrame({"Node": self.nodes}, dtype=object)
        for i in self.nodes:
            fid[f"Fidelity node {i}"] = fid_table[:, i]
        fid.set_index("Node", inplace=True)
        fid.to_csv(self.fid_path)

    def _sha256_block(self, block_path: str):
        """Create a hash from the block csv file."""

        with open(block_path, mode="rb") as block_file:
            content = block_file.read()
            sha256_hash = sha256(content)

        return sha256_hash.hexdigest()

    def state_parameters(self, block_path: str) -> list[float, float]:
        """Create angles of the qubit from the block's hash."""

        sha = self._sha256_block(block_path)
        θ = int(sha[0:32], 16) % (2 * pi)
        ϕ = int(sha[32:], 16) % (pi)

        return [θ, ϕ]
