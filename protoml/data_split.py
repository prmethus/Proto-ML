import stringcolor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def split_data(X, y, mode):
    m = len(X)
    assert len(X) == len(y)

    train_set_size = int(0.8 * len(X))
    if mode == "classification":
        print(
            stringcolor.cs(
                "Using Stratified Sampling for splitting the dataset...", "cyan"
            )
        )
        split = StratifiedShuffleSplit(
            n_splits=1, random_state=0, train_size=train_set_size
        )
        for train_index, validation_index in split.split(X, y):
            X_train, y_train = X[train_index], y[train_index]
            X_validation, y_validation = X[validation_index], y[validation_index]

    elif mode == "regression":
        print(
            stringcolor.cs("Using Random Sampling for splitting the dataset...", "cyan")
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, random_state=0, train_size=train_set_size
        )

    return X_train, y_train, X_validation, y_validation
