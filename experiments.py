"""
experiments.py
--------------
Evaluates the neural network on the WDBC dataset as a function of:
  (i)  network architecture  (number of layers and neurons per layer)
  (ii) regularisation parameter λ

Evaluation strategy
-------------------
Stratified 5-fold cross-validation (k=5).
For each fold we train on 4/5 of the data and evaluate on the remaining 1/5.
Final reported performance = average accuracy and F1 across all 5 folds.
"""

import numpy as np
import pandas as pd
import itertools
import sys
import os
import argparse

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

from neural_network import init_weights, train, predict
from cross_validation import stratified_kfold_split, accuracy, f1_macro
from data_utils import load_digits_dataset, load_parkinsons, load_rice,load_credit, normalize, labels_to_output_matrix,labels_to_onehot, onehot_to_labels,output_matrix_to_labels, load_penguins


#hyperparams
K_FOLDS   = 10        
SEED      = 42 # reproducibility

# Architectures to test- all architectures are written in the report

EXPERIMENTS = [
    # (hidden_sizes,  lam)
    ([8],             0.0),  # shallow, narrow, no reg
    ([16],            0.0),    # shallow, wider, no reg
    ([32],            0.0),    # shallow, widest, no reg
    ([16],            0.1),    # shallow, mid reg
    ([16],            1.0),    # shallow, strong reg
    ([16, 8],         0.0),    # deeper, no reg
]

# Regularisation values to test
# LAMBDAS = [0.0, 0.01, 0.1, 0.25, 1.0]

# Single experiment: one architecture + one lambda, k-fold CV
def run_experiment(X, y, hidden_sizes, lam, n_outputs,
                   alpha=0.5, max_iter=500, k=K_FOLDS, seed=SEED):
    """k-fold CV for one architecture+lambda. Handles binary and multi-class."""
    n_inputs = X.shape[1]
    layer_sizes = [n_inputs] + hidden_sizes + [n_outputs]
    folds = stratified_kfold_split(y, k=k, seed=seed)
    fold_accs, fold_f1s = [], []
 
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = normalize(X[train_idx], X[val_idx])
 
        # Build output matrix: one-hot for multi-class, single col for binary
        if n_outputs > 1:
            Y_train = labels_to_onehot(y[train_idx])
        else:
            Y_train = labels_to_output_matrix(y[train_idx])
 
        thetas = init_weights(layer_sizes, seed=seed + fold_idx)
        trained_thetas, _ = train(thetas, X_train, Y_train,
                                  lam=lam, alpha=alpha, max_iter=max_iter)
 
        # Predict
        raw_preds = predict(trained_thetas, X_val)  # (N_val, n_outputs)
        if n_outputs > 1:
            y_pred = onehot_to_labels(raw_preds)
        else:
            y_pred = output_matrix_to_labels(raw_preds)
 
        fold_accs.append(accuracy(y[val_idx], y_pred))
        fold_f1s.append(f1_macro(y[val_idx], y_pred))
 
    return float(np.mean(fold_accs)), float(np.mean(fold_f1s))

def run_grid_search(dataset_name, data_path=None, output_csv="results.csv"):
    # Load data
    if dataset_name == "digits":
        X, y = load_digits_dataset()
        n_outputs = 10
    elif "parkinson" in dataset_name.lower():
        X, y = load_parkinsons(data_path)
        n_outputs = 1
    elif "rice" in dataset_name.lower():
        X, y = load_rice(data_path)
        n_outputs = 1
    elif "credit" in dataset_name.lower():
        X, y = load_credit(data_path)
        n_outputs = 1
    elif "penguins" in dataset_name.lower():
        X, y = load_penguins(data_path)
        n_outputs = 3 #3 clases
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
 
    print(f"Dataset: {dataset_name}")
    print(f"  {X.shape[0]} instances, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes, {n_outputs} output neuron(s)")
    print(f"Stopping criterion: fixed iterations max_iter=500")
    print(f"Learning rate: alpha=0.5  |  k={K_FOLDS} folds  |  seed={SEED}\n")
 
    rows = []
    total = len(EXPERIMENTS)
    done = 0
 
    for done, (hidden_sizes, lam) in enumerate(EXPERIMENTS):
        arch = str([X.shape[1]] + hidden_sizes + [n_outputs])
        print(f"[{done+1}/{total}] arch={arch}  λ={lam:.3f}  ...", end="", flush=True)
        acc, f1 = run_experiment(X, y, hidden_sizes, lam, n_outputs)
        print(f"  acc={acc:.4f}  f1={f1:.4f}")
        rows.append({"architecture": arch, "lambda": lam,
                     "mean_accuracy": round(acc, 5),
                     "mean_f1": round(f1, 5)})
 
    df = pd.DataFrame(rows).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    print("\n" + "="*60)
    print(df[["architecture", "lambda", "mean_accuracy", "mean_f1"]].to_string())
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    best = df.iloc[0]
    print(f"\nBest: arch={best['architecture']}  λ={best['lambda']}  "
          f"acc={best['mean_accuracy']}  f1={best['mean_f1']}")
    return df
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="'digits' or path to CSV file")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()
 
    if args.data == "digits":
        run_grid_search("digits", output_csv=args.output)
    else:
        run_grid_search(args.data, data_path=args.data, output_csv=args.output)