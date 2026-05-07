"""
learning_curve.py - point 6 
-----------------
how???
- 80-20 split 
- train the network on the first n samples (n = step, 2*step, 3*step …) and then after EACH trainng measure J on full test 
- plot and save as png
Note: take the best architechutre you got from experiments.py and use it here
command line should look like: python learning_curve.py --data wdbc.csv --hidden 8 --lam 0.0
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")         #no display req, sav png omly
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from neural_network import init_weights, train, compute_cost
from cross_validation import stratified_kfold_split
from data_utils import load_loan, load_wdbc, load_titanic, normalize, labels_to_output_matrix, load_adult_income, load_credit, load_digits_dataset, load_parkinsons, load_rice, labels_to_onehot


# Default hyperparameters- can override via cli so its ag 

# DEFAULT_HIDDEN  = [16, 8] #best arch
# DEFAULT_LAMBDA  = 0.01 #best lambda
# DEFAULT_ALPHA   = 0.5       
# DEFAULT_ITERS   = 500       
# DEFAULT_STEP    = 10        
# DEFAULT_SEED    = 42
# DEFAULT_OUTPUT  = "learning_curve.png"

def load_data(data_arg):
    """Load correct dataset and return X, y, n_outputs."""
    d = data_arg.lower()
    if d == "digits":
        X, y = load_digits_dataset();  return X, y, 10
    elif "parkinson" in d:
        X, y = load_parkinsons(data_arg);  return X, y, 1
    elif "rice" in d:
        X, y = load_rice(data_arg);  return X, y, 1
    elif "credit" in d:
        X, y = load_credit(data_arg);  return X, y, 1
    elif "adult" in d or "income" in d:
        X, y = load_adult_income(data_arg);  return X, y, 1
    elif "loan" in d:
        X, y = load_loan(data_arg);  return X, y, 1
    elif "titanic" in d:
        X, y = load_titanic(data_arg);  return X, y, 1
    else:
        X, y = load_wdbc(data_arg);  return X, y, 1

def generate_learning_curve(X, y, n_outputs, hidden_sizes, lam, alpha=0.5, max_iter=500, step=10, seed=42, output_path="learning_curve.png" ):#this si my core func
        X, y,
        # hidden_sizes=DEFAULT_HIDDEN,
        # lam=DEFAULT_LAMBDA,
        # alpha=DEFAULT_ALPHA,
        # max_iter=DEFAULT_ITERS,
        # step=DEFAULT_STEP,
        # seed=DEFAULT_SEED,
        # output_path=DEFAULT_OUTPUT,
        # test_fraction=0.2):
    
        rng = np.random.default_rng(seed)
        #N   = len(X)

        #from cross_validation import stratified_kfold_split
        folds = stratified_kfold_split(y, k=5, seed=seed)
        train_pool_idx, test_idx = folds[0]   # fold 0: train pool + test set

        # Normalise using training-pool statistics only (no leakage)
        X_pool_norm, X_test_norm = normalize(X[train_pool_idx], X[test_idx])
        #Y_test = labels_to_output_matrix(y[test_idx])
        #instead - build test output matrix:
        if n_outputs > 1:
            Y_test = labels_to_onehot(y[test_idx])
        else:
            Y_test = labels_to_output_matrix(y[test_idx])
        #shuffle training order
        shuffled = rng.permutation(len(train_pool_idx))
        X_pool_norm = X_pool_norm[shuffled]
        y_pool  = y[train_pool_idx][shuffled]

        #list of sample sizes
        n_inputs  = X.shape[1]
        #n_outputs = 1- dont need to hardcode 
        layer_sizes = [n_inputs] + hidden_sizes + [n_outputs]

        # Start at min(step, 10) so we always have at least a few samples;
        # end at the full training pool size.
        min_samples = max(step, layer_sizes[1] + 1)  # need > #neurons samples
        sample_sizes = list(range(min_samples, len(train_pool_idx) + 1, step))
        if sample_sizes[-1] != len(train_pool_idx):
            sample_sizes.append(len(train_pool_idx))

        # train on each subset and test on J
        
        print(f"\nGenerating learning curve for  {layer_sizes}")
        print(f"  lambda={lam}, alpha={alpha}, max_iter={max_iter}, step={step}")
        print(f"  Training pool: {len(train_pool_idx)} samples  |  "
            f"Test set: {len(test_idx)} samples")
        print(f"  Evaluating {len(sample_sizes)} sample-size steps ..\n")
        test_costs = []
        for n in sample_sizes:
            X_sub = X_pool_norm[:n]
            if n_outputs > 1:
                Y_sub = labels_to_onehot(y_pool[:n])
            else:
                Y_sub = labels_to_output_matrix(y_pool[:n])
 

            # Fresh weights for each sub-problem (seeded for reproducibility)
            thetas = init_weights(layer_sizes, seed=seed)
            trained_thetas, _ = train(
                thetas, X_sub, Y_sub,
                lam=lam, alpha=alpha, max_iter=max_iter
            )

            # Evaluate J on the fixed test set
            J_test, _, _ = compute_cost(trained_thetas, X_test_norm, Y_test, lam)
            test_costs.append(J_test)

            print(f"  n={n:4d}  J_test={J_test:.5f}")

        # plot
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(sample_sizes, test_costs, color="#2563EB", linewidth=2,
                marker="o", markersize=4, label="Test cost $J$")

        ax.set_xlabel("Number of training samples", fontsize=12)
        ax.set_ylabel("Cost $J$ (cross-entropy + regularisation)", fontsize=12)
        arch_str = str(layer_sizes)
        ax.set_title(
            f"Learning Curve - Architecture {arch_str}\n"
            f"$\\lambda$={lam},  $\\alpha$={alpha},  iterations={max_iter}",
            fontsize=12
        )
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="-", alpha=0.5)
        ax.set_xlim(left=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"\nLearning curve saved to: {output_path}")
        return sample_sizes, test_costs

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--hidden", nargs="+", type=int, default=[16])
    parser.add_argument("--lam",    type=float, default=0.0)
    parser.add_argument("--alpha",  type=float, default=0.5)
    parser.add_argument("--iters",  type=int,   default=500)
    parser.add_argument("--step",   type=int,   default=10)
    parser.add_argument("--output", default="learning_curve.png")
    args = parser.parse_args()
 
    X, y, n_outputs = load_data(args.data)
    generate_learning_curve(
        X, y, n_outputs,
        hidden_sizes=args.hidden,
        lam=args.lam,
        alpha=args.alpha,
        max_iter=args.iters,
        step=args.step,
        output_path=args.output,
    )
 

'''
Note to use: run theses commands: 
python learning_curve.py --data digits --hidden 32 --lam 0.0 --output learning_curve_digits.png
    this format can be used for all datasets- just input the best architechutre you found

'''