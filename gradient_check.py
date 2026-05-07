"""""
Numerically verifies backprop gradients using finite differences.

For each weight:
    formula given in report
    to run just run this file with python gradient_check.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from neural_network import set_weights, compute_cost, backpropagate, _print_matrix


def numerical_gradients(thetas, X, Y, lam, eps):
    num_grads = [np.zeros_like(t) for t in thetas]
    for l, theta in enumerate(thetas):
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                thetas_plus = [t.copy() for t in thetas]
                thetas_plus[l][i, j] += eps
                J_plus, _, _ = compute_cost(thetas_plus, X, Y, lam)

                thetas_minus = [t.copy() for t in thetas]
                thetas_minus[l][i, j] -= eps
                J_minus, _, _ = compute_cost(thetas_minus, X, Y, lam)

                num_grads[l][i, j] = (J_plus - J_minus) / (2.0 * eps)
    return num_grads


def check(thetas, X, Y, lam, eps):
    grads_bp  = backpropagate(thetas, X, Y, lam)
    grads_num = numerical_gradients(thetas, X, Y, lam, eps)

    print(f"\neps = {eps}")
    print("Backprop gradients:")
    for l, g in enumerate(grads_bp):
        print(f"  Theta{l+1}:")
        _print_matrix(g)

    print("Numerical gradients:")
    for l, g in enumerate(grads_num):
        print(f"  Theta{l+1}:")
        _print_matrix(g)

    bp_flat  = np.concatenate([g.flatten() for g in grads_bp])
    num_flat = np.concatenate([g.flatten() for g in grads_num])
    rel_diff = np.linalg.norm(bp_flat - num_flat) / (np.linalg.norm(bp_flat) + np.linalg.norm(num_flat))
    print(f"Relative difference: {rel_diff:.2e}")


# Example 1: [1, 2, 1], lambda=0.0
print("_____________________")
print("Example 1 — network [1, 2, 1], lambda=0.0")
print("_____________________")
thetas1 = set_weights([
    np.array([[0.4, 0.1], [0.3, 0.2]]),
    np.array([[0.7, 0.5, 0.6]])
])
X1 = np.array([[0.13], [0.42]])
Y1 = np.array([[0.90], [0.23]])
check(thetas1, X1, Y1, lam=0.0, eps=0.1)
check(thetas1, X1, Y1, lam=0.0, eps=0.000001)

#Example 2: [2, 4, 3, 2], lambda=0.25
print("_____________________")
print("Example 2 — network [2, 4, 3, 2], lambda=0.25")
print("_____________________")
thetas2 = set_weights([
    np.array([[0.42,0.15,0.40],[0.72,0.10,0.54],[0.01,0.19,0.42],[0.30,0.35,0.68]]),
    np.array([[0.21,0.67,0.14,0.96,0.87],[0.87,0.42,0.20,0.32,0.89],[0.03,0.56,0.80,0.69,0.09]]),
    np.array([[0.04,0.87,0.42,0.53],[0.17,0.10,0.95,0.69]])
])
X2 = np.array([[0.32, 0.68], [0.83, 0.02]])
Y2 = np.array([[0.75, 0.98], [0.75, 0.28]])
check(thetas2, X2, Y2, lam=0.25, eps=0.1)
check(thetas2, X2, Y2, lam=0.25, eps=0.000001)