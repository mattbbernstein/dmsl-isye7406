import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import itertools

def main():
    train, test = read_and_split_data("hw2/fat.csv")
    # full_lr_model = all_features_lr(train)
    # print(test_error(full_lr_model, test))
    best_k_models(train)

def read_and_split_data(filename: str):
    data = pd.read_csv(filename)
    test_rows = [1, 21, 22, 57, 70, 88, 91, 94, 121, 127, 149, 151, 159, 162,
                164, 177, 179, 194, 206, 214, 215, 221, 240, 241, 243]
    train_data = data.drop(test_rows, axis = 0)
    test_data = data.iloc[test_rows]
    return (train_data, test_data)

def all_features_lr(data):
    model = sm.OLS(data.iloc[:,0], data.iloc[:,1:]).fit()
    return model

def best_k_models(data):
    best_features = SelectKBest(score_func=f_regression, k=5).fit(data.iloc[:,1:], data.iloc[:,0])
    print(best_features.get_feature_names_out())
    full_param_search(data.iloc[:,1:], data.iloc[:,0], combo_size=5)

def test_error(model, test_data):
    predictions = model.predict(test_data.iloc[:,1:])
    test_error = np.sum((test_data.iloc[:,0] - predictions) ** 2) / predictions.size
    return test_error

def full_param_search(X, y, combo_size):
    combinations = itertools.combinations(X.columns, r = combo_size)
    best_combo = None
    lowest_score = None
    for combo in combinations:
        this_aic = get_aic(X.loc[:, combo], y)
        if (lowest_score is None) or (this_aic < lowest_score):
            lowest_score = this_aic
            best_combo = combo
    print(best_combo)

def get_aic(X, y):
    model = sm.OLS(y, X).fit()
    return model.aic

if __name__ == "__main__":
    main()