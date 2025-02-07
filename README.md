# PageRank Implementation in Python

This repository contains a Python implementation of the PageRank algorithm using a sparse matrix representation and the power iteration method with a damping factor. 

<img width="1124" alt="graph" src="https://github.com/user-attachments/assets/79ecce34-931d-48c5-9b0d-e50b72e1498d" />

## Features

- **Sparse Matrix Representation:** Constructs the transition matrix as a sparse matrix for memory and computational efficiency.
- **Normalization:** Ensures each column of the transition matrix sums to 1 (stochastic matrix).
- **Power Iteration Method:** Computes the PageRank vector using an iterative approach with convergence tolerance.
- **Damping Factor:** Incorporates the damping factor (set to 0.85) to simulate the behavior of a random surfer.
- **Example Usage:** Includes a simple example with a network of 6 pages and predefined links.

## Requirements

- Python 3.6+
- NumPy
- SciPy


## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/amerob/pagerank-algorithm-python.git
    cd pagerank-algorithm-python
    ```

2. Install dependencies:

    ```bash
    pip3 install -r requirements.txt
    ```
3. Run the script:

    ```bash
    python main.py
    ```

## Note
This code only shows the core concepts behind the PageRank algorithm and serves as a starting point for exploring graph ranking. Contributions are welcome.









