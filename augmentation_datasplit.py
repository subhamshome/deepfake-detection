from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def augmentation_datasplit(X_train_preprocess, X_test_preprocess, y_train, y_test):
    n_comp = 200
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit(X_train_preprocess).transform(X_train_preprocess)
    X_test_pca = pca.transform(X_test_preprocess)

    X_train1, X_validate, y_train1, y_validate = train_test_split(X_train_pca, y_train, test_size=0.2,
                                                                  random_state=15, stratify=y_train, shuffle=True)

    y_train1_ravel = y_train1.values.ravel()
    y_validate_ravel = y_validate.values.ravel()
    y_test_ravel = y_test.values.ravel()

    return X_train1, X_validate, y_train1, y_validate, y_train1_ravel, y_validate_ravel, y_test_ravel, X_test_pca
