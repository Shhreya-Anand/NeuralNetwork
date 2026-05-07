
import numpy as np


# Activation func z- sigmoid

def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))) #added clip to avoid overflow


def sigmoid_derivative(a):
    return a * (1.0 - a)


# Weight initialisation
def init_weights(layer_sizes, seed=None): #return list of weights
    if seed is not None:
        np.random.seed(seed)
    thetas = []
    for l in range(len(layer_sizes) - 1):
        rows = layer_sizes[l + 1]# neurons in next layer
        cols = layer_sizes[l] + 1 # +1 for bias
        theta = np.random.randn(rows, cols) * 0.1 #adds smal random asymmetry
        thetas.append(theta)
    return thetas


def set_weights(thetas_flat):
    return [np.array(t, dtype=float) for t in thetas_flat]

# Forward propagation
def forward_propagate(x, thetas):
    a = np.concatenate([[1.0], x])   # prepend bias to input
    a_list = [a]
    z_list = []

    for l, theta in enumerate(thetas):
        z = theta @ a # (n_next,) = (n_next, n_prev+1) @ (n_prev+1,)
        z_list.append(z)
        a = sigmoid(z)
        if l < len(thetas) - 1:
            a = np.concatenate([[1.0], a])   # prepend bias for hidden layers
        a_list.append(a)

    return a_list, z_list

# Cost function  J
def compute_cost(thetas, X, Y, lam):
    N = len(X)
    total = 0.0
    instance_costs = []
    predictions = []

    for xi, yi in zip(X, Y):
        a_list, _ = forward_propagate(xi, thetas)
        pred = a_list[-1]          # output layer activations (no bias)
        predictions.append(pred)

        # Binary cross-entropy per instance: J function formula is taken from formula in class
        eps = 1e-12                
        j_i = -np.sum(
            yi * np.log(pred + eps) + (1.0 - yi) * np.log(1.0 - pred + eps)
        )
        instance_costs.append(j_i)
        total += j_i

    # Average over instances
    J = total / N
    # Regularisation
    reg = 0.0
    for theta in thetas:
        reg += np.sum(theta[:, 1:] ** 2)   # skip column 0 (bias weights)
    J += (lam / (2.0 * N)) * reg

    return J, instance_costs, predictions

# Backprop:
def backpropagate(thetas, X, Y, lam, verbose=False):
    """

    output gradients - averaged, regularised ready to be used in grad descent
    """
    N = len(X)
    # Accumulator arrays — same shape as each theta
    Delta = [np.zeros_like(t) for t in thetas]

    for i, (xi, yi) in enumerate(zip(X, Y)):
        #Forward pass 
        a_list, z_list = forward_propagate(xi, thetas)

        if verbose: #boolean
            print(f"\n\tComputing gradients based on training instance {i+1}")
            print(f"\t\tForward propagating the input {_fmt(xi)}")
            for l, a in enumerate(a_list):
                print(f"\t\ta{l+1}: {_fmt(a)}")
                if l < len(z_list):
                    print(f"\t\tz{l+2}: {_fmt(z_list[l])}")

        # baward pass
        deltas = [None] * len(thetas)
        pred = a_list[-1]
        deltas[-1] = pred - yi   # shape: (n_outputs,)

        # using formuila for delta for hidden layers (lecture vid)
        for l in range(len(thetas) - 2, -1, -1):
            # a_list[l+1] includes a prepended bias for hidden layers
            # The actual activations (without bias) start at index 1
            a_hidden = a_list[l + 1][1:]   # strip bias
            # Propagate through weights (skip bias column — col 0)
            delta_prop = thetas[l + 1][:, 1:].T @ deltas[l + 1]
            deltas[l] = delta_prop * sigmoid_derivative(a_hidden)

        if verbose:
            for l, d in enumerate(deltas):
                print(f"\t\tdelta{l+2}: {_fmt(d)}")

        for l in range(len(thetas)): #bruh accumulate
            # a_list[l] has the bias unit prepended for all but the last layer
            grad_i = np.outer(deltas[l], a_list[l])
            Delta[l] += grad_i

            if verbose:
                print(f"\n\t\tGradients of Theta{l+1} based on training instance {i+1}:")
                _print_matrix(grad_i)

    #Average + regularise
    gradients = []
    for l, (acc, theta) in enumerate(zip(Delta, thetas)):
        g = acc / N
        # Add regularisation to non-bias weights
        reg_mask = np.zeros_like(theta)
        reg_mask[:, 1:] = theta[:, 1:]   # only non-bias weights
        g += (lam / N) * reg_mask
        gradients.append(g)

    if verbose:
        print("\n\tThe entire training set has been processed. "
              "Computing avg (regularized) gradients:")
        for l, g in enumerate(gradients):
            print(f"\t\tFinal regularized gradients of Theta{l+1}:")
            _print_matrix(g)

    return gradients

# Weight update (gradient descent)
def update_weights(thetas, gradients, alpha):
    """Standard gradient descent step: θ <- θ - alpha * del (J)"""
    return [t - alpha * g for t, g in zip(thetas, gradients)]

def train(thetas, X, Y, lam, alpha=0.1, max_iter=1000, epsilon=None,
          verbose=False):
    costs = []
    for iteration in range(max_iter):
        J, _, _ = compute_cost(thetas, X, Y, lam)
        costs.append(J)

        grads = backpropagate(thetas, X, Y, lam, verbose=verbose)
        thetas = update_weights(thetas, grads, alpha)

        if epsilon is not None and len(costs) > 1:
            if abs(costs[-2] - costs[-1]) < epsilon:
                break

    return thetas, costs
# Prediction helpers
def predict(thetas, X): #return pred class labels (0,1)
    preds = []
    for xi in X:
        a_list, _ = forward_propagate(xi, thetas)
        preds.append((a_list[-1] >= 0.5).astype(int))
    return np.array(preds)

#these are all for verbose mode
def _fmt(arr): #formating 1d array like reference files
    return "[" + "   ".join(f"{v:.5f}" for v in np.atleast_1d(arr)) + "]"
def _print_matrix(mat):
    for row in mat:
        print("\t\t\t" + "  ".join(f"{v:.5f}" for v in row))
