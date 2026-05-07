"""from scrath
"""

import numpy as np


def stratified_kfold_split(y, k=5, seed=None):
    """
    returns folds-)list of k tuples (train_idx, val_idx)
    -Both are 1D NumPy integer arrays.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)

    # Collect indices for each unique class
    classes = np.unique(y)
    class_indices = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx) # in-place shuffle within this class
        class_indices[c] = idx

    # Build k bins per class, then assemble each fold
    # bins[c][f] = indices of class c that belong to fold f
    bins = {c: np.array_split(idx, k) for c, idx in class_indices.items()}

    folds = []
    for f in range(k):
        val_idx = np.concatenate([bins[c][f] for c in classes])
        train_idx = np.concatenate(
            [bins[c][j] for c in classes for j in range(k) if j != f]
        )
        # Shuffle so the returned indices are not sorted by class
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)
        folds.append((train_idx, val_idx))

    return folds
#acc, f1

def accuracy(y_true, y_pred):
    """correctly classified instances/total"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.mean(y_true == y_pred)


def f1_score_binary(y_true, y_pred, positive_class=1):
    """
    f1 = 2 * precision * recall / (precision + recall)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
    fn = np.sum((y_pred != positive_class) & (y_true == positive_class))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_macro(y_true, y_pred): #this is avg f1 across all classes- just average them, np.mean
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = [f1_score_binary(y_true, y_pred, c) for c in classes]
    return float(np.mean(scores))
