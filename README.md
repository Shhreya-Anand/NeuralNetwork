# Neural Network — Backpropagation Assignment

## File structure

```
neural_net/
├── neural_network.py     # Core: forward prop, backprop, cost, training loop
├── verify_backprop.py    # Correctness check against reference output files
├── cross_validation.py   # Stratified k-fold CV, accuracy, F1 metrics
├── data_utils.py         # WDBC data loading and min-max normalisation
├── experiments.py        # Grid search: architectures × λ values
└── main.py               # Top-level entry point (all modes)
```

---

## Quick start

### 1. Verify backpropagation correctness
Reproduces all intermediate quantities (z, a, δ, gradients) for both
reference files and prints them to stdout.

```bash
# Both examples
python main.py --mode verify

# Or individually
python verify_backprop.py --example 1
python verify_backprop.py --example 2
```

Compare the printed output with `backprop_example1.txt` and
`backprop_example2.txt`.  Every value should match to 5 decimal places.

### 2. Run WDBC experiments
```bash
python main.py --mode train --data wdbc.csv --output results_wdbc.csv
```

Or run both verification + experiments in one command:
```bash
python main.py --data wdbc.csv
```

---

## Module documentation

### `neural_network.py`
| Function | Purpose |
|---|---|
| `sigmoid(z)` | Element-wise sigmoid activation σ(z) = 1/(1+e⁻ᶻ) |
| `sigmoid_derivative(a)` | σ'(z) = a(1−a) given already-activated value |
| `init_weights(layer_sizes)` | Random weight init; rows = neurons, col 0 = bias |
| `set_weights(thetas_flat)` | Load exact weights (used for verification) |
| `forward_propagate(x, thetas)` | Returns a_list and z_list for one input |
| `compute_cost(thetas, X, Y, lam)` | Regularised cross-entropy cost J |
| `backpropagate(thetas, X, Y, lam)` | Full backprop; returns averaged regularised gradients |
| `update_weights(thetas, grads, alpha)` | θ ← θ − α∇J |
| `train(...)` | Full training loop; stops at max_iter or Δε |
| `predict(thetas, X)` | Returns 0/1 predictions |

**Weight layout:** `Theta[l]` has shape `(neurons_next, neurons_current + 1)`.
Row `i` = weights incoming to neuron `i` in the next layer.
Column 0 = bias weight (never regularised).

### `verify_backprop.py`
Hard-codes the exact weights and data from both reference files, runs
forward propagation and backpropagation, and prints every intermediate
quantity in the same format as the reference files.

### `cross_validation.py`
| Function | Purpose |
|---|---|
| `stratified_kfold_split(y, k)` | Returns k (train_idx, val_idx) pairs with matched class ratios |
| `accuracy(y_true, y_pred)` | Fraction correct |
| `f1_score_binary(y_true, y_pred, pos)` | F1 for one class |
| `f1_macro(y_true, y_pred)` | Macro-averaged F1 across all classes |

### `data_utils.py`
| Function | Purpose |
|---|---|
| `load_wdbc(path)` | Returns `(X, y)` from wdbc.csv |
| `normalize(X_train, X_val)` | Min-max normalisation using training stats only |
| `labels_to_output_matrix(y)` | Converts 1-D labels to (N,1) for the network |
| `output_matrix_to_labels(Y_pred)` | Converts (N,1) predictions to 1-D labels |

### `experiments.py`
Grid-searches over `HIDDEN_ARCHITECTURES × LAMBDAS` using 5-fold
stratified CV.  Stopping criterion: **fixed iterations** (`max_iter=500`,
`alpha=0.5`).  Results are saved to `results_wdbc.csv`.

---

## Design decisions

- **Sigmoid everywhere:** matches the reference files; no ReLU or softmax.
- **Cross-entropy loss:** standard for binary classification with sigmoid outputs.
- **Regularisation:** λ penalty on all non-bias weights, averaged over N instances.
- **Stopping criterion:** fixed iterations (max_iter).  Simple and reproducible.
  You can switch to ε-stopping by passing `epsilon=1e-5` to `train()`.
- **Normalisation:** min-max per fold, computed only on training data, applied to val.
- **No external ML libraries** used for the neural network logic (only NumPy and pandas).
