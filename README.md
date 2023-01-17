# QAOA based classifier for detecting credit card fraud

## Abstract
Fraud detection in credit card transactions is a major challenge in machine learning. Instead of resorting to classical algorithms, we build a Quantum Machine Learning model based off QAOA circuits. We create a classifier using this ansatz and optimize it in order to guess whether a transaction is legit or fraudulent. The project is implemented using the framework for QML Pennylane.


How it works? The algorithm has four steps:

1. QAOA ansatz: build a quantum circuit based off QAOA circuit [1].
2. Encoding: encode the class into the expected value of the Hamiltonian Hc.
2. Cost function: Create a cost function whose minimum is reached when all predictions are on point.
3. Training: Train the circuit using a classical optimizer.

## Organization
The project is organized as follows
1. The main script to be run in the Docker container is ``main.py``.
2. The findings, description of the process and results are reported in the jupyter notebook ``main_notebook.ipynb``.
3. The main module of the quantum classifier is found in ``Classifier/quantum_classifier.py``.
4. The database is stored in the ``Data`` folder.

### Problem

The program works untill the optimization step. It apparently does not detect that the parameters have ```require_grad = True``` turned on. It then gives an ```IndexError: tuple index out of range``` in the grad function of the Pennylane library. In the future, I will investigate the issue and try to fix it...

[1] Farhi, Goldstone, Gutmann, "A Quantum Approximate Optimization Algorithm" (2014) arXiv:1411.4028