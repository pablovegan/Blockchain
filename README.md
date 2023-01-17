# Quantum Blockchain in Python

***Important note:*** *this is a toy model of a blockchain we made for the Qiskit Global Hackathon organized by IBM. It doesn't actually solve any of the inherent problems of the blockchain, so it is not of particular use in the real world (it is not even clear blockchain has any useful applications). It was nonetheless an interesting project to work on and it encouraged us to learn a lot about blockchain.*

 We use quantum information and computation in the blockchain in four ways: we encode the blocks 'hash' state in a qubit, we send qubits using quantum teleportation, we generate random numbers using an entangled circuit and finally we select the winner in each block solving round measuring fidelities between states.

## Class node
Starts by initializing the number of nodes, an index to identify them and the first transactions paths and transaction count.
It contains the *transaction function* which verifies that the transaction can be made by checking that the funds in the wallet are greater than the transaction to be made. If true, rewrites the wallet of the involving nodes.
The *create_block* function creates a text file containing the transaction of the previous block, the previous state, transactions and the block number. It stores the information of the winning node and sets the payments for solving the blocks and the maintenance reward. In addition, it stores the information of the previous fidelities.
*Update_winners* function sets who solves the block by obtaining the major fidelity. On the other hand, *write_fidelitiy* saves the fidelity table in a cvs file.
*State_ parameters* function hashes the block information, thanks to the *Sha256*  function, in two parts obtaining two parameters. To represent these parameters as a state in the block sphere it brings its values to (0,π) and (0,2π) obtaining this way θ, ϕ, which determines only one state.

## Class blockchain
This class creates a new blockchain. First, parameters which describe the blockchain are initialized: number of nodes, allowed transactions per block, , and the length of the random numbers involved. After parameters, attributes are deffined. Appart from the previously mentioned, a list of the nodes indices and a list of the nodes in the chain are provided.
Then, three .csv files are created for each node in order to record transactions, fidelities and blocks.

The corresponding functions are described as follows: 
*initialize_files* a initial transactions file is inizialited giving each node 20 coins to make transactions.
*random_numbers* creates as many sequences of random numbers as nodes by means of *random_digits*. Those numbers are generated using a quantum generator and are used to verify the block at the end of the process. 
*send_states* is used to simulate the process of sending a state between two nodes using quantum teleportation.
*fidelity_table* simulates the process of measuring all possible fidelities between nodes.
*get_winners* search the maximum fidelities between all the calculated ones.
*update_blocks* tells the nodes about the winner node in order to get the blocks updapted. Also, gives the reward to the winners.
*solve_block* verifies the blocks of the winners by means of the calculation of fidelities and the random numbers
*add_transactions* enables user to add to the blockchain a particular transaction specifying nodes and the amount.