"""
to do: 
1. Separate X features and y labels and normalise to 0,1, convert bin col to output matrix - check the shape
2. Min-max normalise every feature to [0, 1].
"""

import numpy as np
import pandas as pd


def load_wdbc(path):
    
    df = pd.read_csv(path)
    # Last column is the label; all others are features
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y


def normalize(X_train, X_val=None):
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0   # avoid divide-by-zero for constant columns

    X_train_norm = (X_train - mins) / ranges

    if X_val is not None:
        X_val_norm = (X_val - mins) / ranges
        # Clip to handle rare cases where a val value falls slightly outside
        # the training range
        X_val_norm = np.clip(X_val_norm, 0.0, 1.0)
        return X_train_norm, X_val_norm

    return X_train_norm, None


def labels_to_output_matrix(y):
    return y.reshape(-1, 1).astype(float)#NN has output of 1 neuron


def output_matrix_to_labels(Y_pred):
    return (Y_pred[:, 0] >= 0.5).astype(int)

def load_loan(path): #one-hot encoding 
    df = pd.read_csv(path)
 
    # Columns to one-hot encode (string categoricals)
    # attr10_cat is already 0/1 so we leave it as a number
    cat_cols = ['attr1_cat', 'attr2_cat', 'attr3_cat',
                'attr4_cat', 'attr5_cat', 'attr11_cat']
    num_cols = ['attr6_num', 'attr7_num', 'attr8_num',
                'attr9_num', 'attr10_cat']
 
    # One-hot encode categorical columns.
    # drop_first=False keeps all k dummies so no information is lost.
    # dtype=float ensures the output is numeric.
    dummies = pd.get_dummies(df[cat_cols], drop_first=False, dtype=float)
 
    # Combine with numerical columns
    X_df = pd.concat([dummies, df[num_cols].astype(float)], axis=1)
    X = X_df.values.astype(float)
    y = df["label"].values.astype(int)
 
    return X, y
 
'''
TITANIC: 
Note: only  attr2_cat (male/female) needs one hot encoding, attr1_cat is alr integer type. Write similar logicc to load_loan ok cool
'''

def load_titanic(path):
    df = pd.read_csv(path)
 
    # One-hot encode the only string categorical column
    dummies = pd.get_dummies(df[['attr2_cat']], drop_first=False, dtype=float)
 
    # Combine with integer/numerical columns
    num_cols = ['attr1_cat', 'attr3_num', 'attr4_num', 'attr5_num', 'attr6_num']
    X_df = pd.concat([dummies, df[num_cols].astype(float)], axis=1)
    X = X_df.values.astype(float)
    y = df['label'].values.astype(int)
 
    return X, y

def load_digits_dataset():
    """Load the sklearn hand-written digits dataset.
 
    10 classes (0-9), 1797 instances, 64 numerical features.
    Labels are integers 0-9 — multi-class, not binary.
 
    Returns
    -------
    X : (1797, 64) float array
    y : (1797,)    int array  — values 0 through 9
    """
    from sklearn import datasets
    digits = datasets.load_digits(return_X_y=True)
    X = digits[0].astype(float)
    y = digits[1].astype(int)
    return X, y
 
 
def load_parkinsons(path):
    """Load the Parkinson's disease dataset.
 
    All 22 attributes are numerical. Label column is 'Diagnosis'.
    0 = healthy, 1 = Parkinson's.
 
    Returns
    -------
    X : (195, 22) float array
    y : (195,)    int array
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != 'Diagnosis']
    X = df[feature_cols].values.astype(float)
    y = df['Diagnosis'].values.astype(int)
    return X, y
 
 
def load_rice(path):
    """Load the Rice Grains dataset.
 
    Same structure as Raisin — all numerical, label column is 'label'.
 
    Returns
    -------
    X : (3810, 7) float array
    y : (3810,)   int array
    """
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values.astype(float)
    classes = sorted(df['label'].unique())
    y = df['label'].map({c: i for i, c in enumerate(classes)}).values.astype(int)
    return X, y
 
 
def load_credit(path):
    """Load the Credit Approval dataset.
 
    9 categorical columns are one-hot encoded.
    6 numerical columns kept as-is.
 
    Returns
    -------
    X : (653, n_features) float array
    y : (653,)            int array
    """
    df = pd.read_csv(path)
    cat_cols = ['attr1_cat', 'attr4_cat', 'attr5_cat', 'attr6_cat',
                'attr7_cat', 'attr9_cat', 'attr10_cat', 'attr12_cat', 'attr13_cat']
    num_cols = ['attr2_num', 'attr3_num', 'attr8_num',
                'attr11_cat', 'attr14_num', 'attr15_num']
    dummies = pd.get_dummies(df[cat_cols], drop_first=False, dtype=float)
    X_df = pd.concat([dummies, df[num_cols].astype(float)], axis=1)
    X = X_df.values.astype(float)
    y = df['label'].values.astype(int)
    return X, y
 
 
def labels_to_onehot(y):
    """Convert integer labels to one-hot matrix for multi-class output.
 
    E.g. y=3 with 10 classes -> [0,0,0,1,0,0,0,0,0,0]
 
    Used for digits dataset (10 classes).
    """
    classes = np.unique(y)
    n_classes = len(classes)
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i, label] = 1.0
    return Y
 
 
def onehot_to_labels(Y_pred):
    """Convert one-hot predicted output matrix to integer labels.
 
    Takes argmax across columns — picks the neuron with highest activation.
    """
    return np.argmax(Y_pred, axis=1)

'''New dataset: adult income - preprocessing fto ranodmly sample 3000 instances
- Drops 'native-country' (42 unique values -> too many one-hot cols)
    and 'fnlwgt' (census weight, not a predictive feature)
-Categorical columns one-hot encoded:
    workclass (9), education (16), marital-status (7), occupation (15), relationship (6), race (5), gender(2)
-Numerical columns kept as-is: age, educational-num, capital-gain, capital-loss, hours-per-week
- Label: income -> 0 (<=50K), 1 (>50K)

returns X: (n_sampels, n_features)- float array ; y: (n_samples) - int arr

'''
def load_adult_income(path, n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(path)
 
    # Encode label
    df['label'] = (df['income'] == '>50K').astype(int)
 
    # Drop unhelpful columns
    df = df.drop(columns=['income', 'fnlwgt', 'native-country'])
 
    # Stratified subsample
    class0 = df[df['label'] == 0]
    class1 = df[df['label'] == 1]
    # Keep class ratio: ~76% class0, ~24% class1
    n0 = int(n_samples * len(class0) / len(df))
    n1 = n_samples - n0
    idx0 = rng.choice(len(class0), n0, replace=False)
    idx1 = rng.choice(len(class1), n1, replace=False)
    df = pd.concat([class0.iloc[idx0], class1.iloc[idx1]]).reset_index(drop=True)
 
    # One-hot encode categorical columns
    cat_cols = ['workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'gender']
    num_cols = ['age', 'educational-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
 
    dummies = pd.get_dummies(df[cat_cols], drop_first=False, dtype=float)
    X_df = pd.concat([dummies, df[num_cols].astype(float)], axis=1)
    X = X_df.values.astype(float)
    y = df['label'].values.astype(int)

    # Save preprocessed dataset- remeber to send it to the rest
    out_df = pd.DataFrame(X_df)
    out_df['label'] = y
    out_df.to_csv('newadultincome.csv', index=False)
 
    return X, y