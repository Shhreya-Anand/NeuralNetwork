# Neural Network: Backprop. forward prop, training

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
- evaluated on performance of k=10 fold cross validation
---


## Things to worry about/how to run: 
- all data is already imported here, Note: newadultincome is used for 5th dataset (stratified random sampling of original adultincome.csv dataset)

files to care about: **experiments.py** , **data_utils.py**, **learning_curve.py** 

how to run the commands for each of these files is already given as a note ontop of each of these files. 


---

## Design decisions

- **Sigmoid everywhere:** matches the reference files; no ReLU or softmax.
- **Cross-entropy loss:** standard for binary classification with sigmoid outputs.
- **Regularisation:** λ penalty on all non-bias weights, averaged over N instances.
- **Stopping criterion:** fixed iterations (max_iter).  Simple and reproducible.
  You can switch to ε-stopping by passing `epsilon=1e-5` to `train()`.
- **Normalisation:** min-max per fold, computed only on training data, applied to val.
- **No external ML libraries** used for the neural network logic (only NumPy and pandas).
