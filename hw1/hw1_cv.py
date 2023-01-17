import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

SEED = 7406
N_ROUNDS = 100

def main():
    rand_state = np.random.RandomState(SEED)

    train, test = read_data()
    n_test = test.shape[0]
    data = pd.concat([train, test])
    cols = ["LinearRegression", "KNN1", "KNN3", "KNN5", "KNN7", "KNN9", "KNN11", "KNN13", "KNN15"]
    test_err = pd.DataFrame(None, columns=cols)
    for i in range(N_ROUNDS):
        this_row = []
        train_X, test_X, train_y, test_y = montecarlo_split(data, n_test, rand_state)
        lin_error = linear_test_error(train_X, train_y, test_X, test_y)
        this_row.append(lin_error)
        k_choices = list(range(1, 16, 2))
        for k in k_choices:
            knn_error = knn_test_error(k, train_X, train_y, test_X, test_y)
            this_row.append(knn_error)
        this_row = pd.DataFrame([this_row], columns=cols)
        test_err = pd.concat([test_err, this_row])

    summary = test_err.describe()
    summary = summary.apply(lambda x: np.square(x) if x.name == 'std' else x, axis = 1)
    summary = summary.rename(index={"std":"var"})
    print(summary)

def read_data() -> tuple:
    zip_train = pd.read_csv("hw1/zip.train.csv", header=None)
    zip_test = pd.read_csv("hw1/zip.test.csv", header=None)

    col_names = ["Y"]
    col_names.extend(["X%d" % i for i in range(zip_train.shape[1]-1)])
    zip_train.columns = col_names
    zip_test.columns = col_names
    zip_train = zip_train[zip_train["Y"].isin([2, 7])]
    zip_test = zip_test[zip_test["Y"].isin([2, 7])]


    return (zip_train, zip_test)

def montecarlo_split(data, n_test = -1, rand_state = None) -> tuple:
    y = data["Y"]
    X = data.drop(columns = ["Y"])

    if n_test == -1:
        n_test = np.floor(data.shape[0] * 0.3)

    splits = train_test_split(X, y, test_size=n_test, random_state=rand_state)
    return splits

def linear_test_error(train_X, train_y, test_X, test_y) -> float:
    model = LinearRegression().fit(train_X, train_y)
    pred = model.predict(test_X)
    pred = 2 + 5 * (pred >= 4.5)
    err = np.mean(pred != test_y)
    return err

def knn_test_error(k, train_X, train_y, test_X, test_y) -> float:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X, train_y)
    pred = knn.predict(test_X)
    err = np.mean(pred != test_y)
    return err


if __name__ == "__main__":
    main()