import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def nan_handler(X_train, X_test):
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train = X_train.dropna(axis='columns')

    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test = X_test.dropna(axis='columns')

    train_cols = X_train.columns
    test_cols = X_test.columns

    train_not_test = train_cols.difference(test_cols)
    test_not_train = test_cols.difference(train_cols)
    if len(train_not_test) > len(test_not_train):
        X_train.drop(list(train_not_test), axis=1, inplace=True)
    elif len(train_not_test) < len(test_not_train):
        X_test.drop(list(test_not_train), axis=1, inplace=True)


def scaler_imputer(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    imputer = SimpleImputer(strategy="mean")
    X_train_preprocess = imputer.fit_transform(X_train_scaled)
    X_test_preprocess = imputer.transform(X_test_scaled)

    return X_train_preprocess, X_test_preprocess
