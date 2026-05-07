"""
verify_backprop.py
------------------
Reproduces the step-by-step output described in backprop_example1.txt and
backprop_example2.txt so you can confirm the implementation is correct.

how to?
1. hard-code the exact network architecture, init\ weights, and training
   data described in each reference file (rather than parsing the files)
2. run forward prop, compute J, then run backpropagation, printing
   every intermediate quantity
3. The printed numbers should match the reference files 

"""

import argparse
import numpy as np
from neural_network import (
    set_weights, forward_propagate, compute_cost,
    backpropagate, _fmt, _print_matrix
)


def _print_forward_and_cost(thetas, X, Y, lam):
    """to do: 
    1. print forward prop section, per instance cost
    2. regularized cost J"""

    print("--------------------------------------------")
    print("Computing the error/cost, J, of the network")

    N = len(X)
    total_j = 0.0
    instance_costs = []

    for i, (xi, yi) in enumerate(zip(X, Y)):
        print(f"\tProcessing training instance {i+1}")
        print(f"\tForward propagating the input {_fmt(xi)}")

        a_list, z_list = forward_propagate(xi, thetas)

        # TO PRint each layers z and a's - use for loop
        for l in range(len(thetas)):
            print(f"\t\ta{l+1}: {_fmt(a_list[l])}")
            print()
            print(f"\t\tz{l+2}: {_fmt(z_list[l])}")
            print(f"\t\ta{l+2}: {_fmt(a_list[l+1])}")
            print()

        pred = a_list[-1]
        eps = 1e-12
        j_i = -np.sum(yi * np.log(pred + eps) + (1 - yi) * np.log(1 - pred + eps))
        instance_costs.append(j_i)
        total_j += j_i

        print(f"\t\tf(x): {_fmt(pred)}")
        print(f"\tPredicted output for instance {i+1}: {_fmt(pred)}")
        print(f"\tExpected output for instance {i+1}: {_fmt(yi)}")
        print(f"\tCost, J, associated with instance {i+1}: {j_i:.3f}")
        print()

    # Average + regularisation
    J = total_j / N
    reg = sum(np.sum(t[:, 1:] ** 2) for t in thetas)
    J += (lam / (2 * N)) * reg
    print(f"Final (regularized) cost, J, based on the complete training set: {J:.5f}")
    print()


def _print_backprop(thetas, X, Y, lam):
    """to do: 
    1. print backprop with per instance delta values
    2. per instance gradients, 
    3. avg regularized gradients
    """

    print("--------------------------------------------")
    print("Running backpropagation")

    N = len(X)
    Delta = [np.zeros_like(t) for t in thetas]

    for i, (xi, yi) in enumerate(zip(X, Y)):
        print(f"\tComputing gradients based on training instance {i+1}")

        a_list, z_list = forward_propagate(xi, thetas)
        pred = a_list[-1]

        # Output layer delta
        deltas = [None] * len(thetas)
        deltas[-1] = pred - yi
        print(f"\t\tdelta{len(thetas)+1}: {_fmt(deltas[-1])}")

        # Hidden layer deltas (back to front)
        for l in range(len(thetas) - 2, -1, -1):
            a_hidden = a_list[l + 1][1:]      # strip bias
            delta_prop = thetas[l + 1][:, 1:].T @ deltas[l + 1]
            deltas[l] = delta_prop * a_hidden * (1 - a_hidden)
            print(f"\t\tdelta{l+2}: {_fmt(deltas[l])}")

        print()

        # Per-instance gradients and accumulation
        for l in range(len(thetas)):
            grad_i = np.outer(deltas[l], a_list[l])
            Delta[l] += grad_i
            print(f"\t\tGradients of Theta{l+1} based on training instance {i+1}:")
            _print_matrix(grad_i)
            print()

    # Final averaged regularised gradients
    print("\tThe entire training set has been processed. "
          "Computing the average (regularized) gradients:")
    for l, (acc, theta) in enumerate(zip(Delta, thetas)):
        g = acc / N
        reg_mask = np.zeros_like(theta)
        reg_mask[:, 1:] = theta[:, 1:]
        g += (lam / N) * reg_mask
        print(f"\t\tFinal regularized gradients of Theta{l+1}:")
        _print_matrix(g)
        print()



# Example 1:  network [1, 2, 1], λ=0.0


def run_example1():
    print("=" * 60)
    print("EXAMPLE 1  —  network [1, 2, 1],  λ = 0.000")
    print("=" * 60)

    lam = 0.0

    # Initial weights exactly as given in backprop_example1.txt
    # Theta1: 2 neurons in hidden layer, each with bias + 1 input weight
    # Row i = weights of neuron i (bias weight first)
    Theta1 = np.array([
        [0.4, 0.1],
        [0.3, 0.2],
    ])
    # Theta2: 1 output neuron, bias + 2 hidden weights
    Theta2 = np.array([
        [0.7, 0.5, 0.6],
    ])
    thetas = set_weights([Theta1, Theta2])

    # Training data
    X = np.array([[0.13], [0.42]])
    Y = np.array([[0.90], [0.23]])

    print(f"\nRegularization parameter lambda={lam:.3f}\n")
    print("Initializing the network with the following structure "
          "(number of neurons per layer): [1 2 1]\n")
    print("Initial Theta1:")
    _print_matrix(Theta1)
    print("\nInitial Theta2:")
    _print_matrix(Theta2)
    print()

    print("Training set")
    for i, (xi, yi) in enumerate(zip(X, Y)):
        print(f"\tTraining instance {i+1}")
        print(f"\t\tx: {_fmt(xi)}")
        print(f"\t\ty: {_fmt(yi)}")
    print()

    _print_forward_and_cost(thetas, X, Y, lam)
    _print_backprop(thetas, X, Y, lam)


# Example 2:  network [2, 4, 3, 2], λ=0.25


def run_example2():
    print("=" * 60)
    print("EXAMPLE 2  —  network [2, 4, 3, 2],  λ = 0.250")
    print("=" * 60)

    lam = 0.25

    # Initial weights exactly as given in backprop_example2.txt
    Theta1 = np.array([
        [0.42, 0.15, 0.40],
        [0.72, 0.10, 0.54],
        [0.01, 0.19, 0.42],
        [0.30, 0.35, 0.68],
    ])
    Theta2 = np.array([
        [0.21, 0.67, 0.14, 0.96, 0.87],
        [0.87, 0.42, 0.20, 0.32, 0.89],
        [0.03, 0.56, 0.80, 0.69, 0.09],
    ])
    Theta3 = np.array([
        [0.04, 0.87, 0.42, 0.53],
        [0.17, 0.10, 0.95, 0.69],
    ])
    thetas = set_weights([Theta1, Theta2, Theta3])

    # Training data
    X = np.array([[0.32, 0.68], [0.83, 0.02]])
    Y = np.array([[0.75, 0.98], [0.75, 0.28]])

    print(f"\nRegularization parameter lambda={lam:.3f}\n")
    print("Initializing the network with the following structure "
          "(number of neurons per layer): [2 4 3 2]\n")
    print("Initial Theta1:")
    _print_matrix(Theta1)
    print("\nInitial Theta2:")
    _print_matrix(Theta2)
    print("\nInitial Theta3:")
    _print_matrix(Theta3)
    print()

    print("Training set")
    for i, (xi, yi) in enumerate(zip(X, Y)):
        print(f"\tTraining instance {i+1}")
        print(f"\t\tx: {_fmt(xi)}")
        print(f"\t\ty: {_fmt(yi)}")
    print()

    _print_forward_and_cost(thetas, X, Y, lam)
    _print_backprop(thetas, X, Y, lam)



# Entry point- main func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifying backpropagainst reference output files.")
    parser.add_argument("--example", type=int, choices=[1, 2],
                        help="Which example to run (default: both)")
    args = parser.parse_args()

    if args.example == 1:
        run_example1()
    elif args.example == 2:
        run_example2()
    else:
        run_example1()
        print("\n\n")
        run_example2()
