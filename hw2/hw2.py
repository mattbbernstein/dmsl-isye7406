import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import itertools

SEED = 7406
RAND_STATE = np.random.RandomState(SEED)

def main():
    train, test = read_and_split_data("hw2/fat.csv")
    model_names = ["Full", "5-Best (p-score)", "5-Best (AIC)", "Stepwise", "LASSO", "Ridge", "PCA", "PLS"]
    # full_lr_model = all_features_lr(train)
    # k_p_model, k_aic_model = best_k_models(train)
    # lasso_model, lasso_lambda = lasso(train, plot=False)
    # ridge_model, ridge_lambda, ridge_scaler = ridge(train)
    pca_model, pca_trans, pca_scaler = pca_regr(train, plot=False)

    n_pcs = pca_model.n_features_in_
    scaled_test = pca_scaler.transform(test)
    pca_test = pca_trans.transform(scaled_test[:,1:])[:,:n_pcs]
    pca_predictions = pca_model.predict(pca_test)
    pca_mse = np.sum((test.iloc[:,0] - pca_predictions) ** 2) / pca_predictions.size
    print(pca_mse)

def read_and_split_data(filename: str):
    data = pd.read_csv(filename)
    data = data.drop(columns = ["siri", "density", "free"])
    test_rows = [1, 21, 22, 57, 70, 88, 91, 94, 121, 127, 149, 151, 159, 162,
                164, 177, 179, 194, 206, 214, 215, 221, 240, 241, 243]
    train_data = data.drop(test_rows, axis = 0)
    test_data = data.iloc[test_rows]
    return (train_data, test_data)

def all_features_lr(data):
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    model = sm.OLS(y, X).fit()
    return model

def best_k_models(data):
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    p_best_features = SelectKBest(score_func=f_regression, k=5).fit(X, y)
    model_pscore = sm.OLS(y, data.loc[:, p_best_features.get_feature_names_out()]).fit()
    aic_best_features = full_param_search(X, y, combo_size=5)
    model_aic = sm.OLS(y, data.loc[:, aic_best_features]).fit()
    return (model_pscore, model_aic)

def full_param_search(X, y, combo_size):
    combinations = itertools.combinations(X.columns, r = combo_size)
    best_combo = None
    lowest_score = None
    for combo in combinations:
        this_aic = get_aic(X.loc[:, combo], y)
        if (lowest_score is None) or (this_aic < lowest_score):
            lowest_score = this_aic
            best_combo = combo
    return best_combo

def lasso(data, plot = False):
    scaled_data = StandardScaler().fit_transform(data)
    scaled_X = scaled_data[:,1:]
    y = data.iloc[:,0]
    lasso_cv = LassoCV(cv=5, random_state=RAND_STATE).fit(scaled_X, y)
    if plot:
        lambdas, coefs, _ = lasso_cv.path(scaled_X,y, alphas = lasso_cv.alphas_)
        plot_coef_path(lambdas, coefs, "LASSO", best_lambda = lasso_cv.alpha_)
    selected_coefficients = pd.DataFrame(list(zip(data.columns[1:], lasso_cv.coef_)), columns = ["Feature", "Coefficient"])
    selected_coefficients = selected_coefficients[selected_coefficients["Coefficient"] != 0]
    
    X = data.loc[:,selected_coefficients["Feature"]]
    model = sm.OLS(y, X).fit()
    return (model, lasso_cv.alpha_)

def ridge(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    X = scaled_data[:,1:]
    y = data.iloc[:,0]
    ridge_cv = RidgeCV(cv=5).fit(X, y)
    return (ridge_cv, ridge_cv.alpha_, scaler)
    
def pca_regr(data, plot = False):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_X = scaled_data[:,1:]
    y = data.iloc[:,0]
    pca = PCA()
    red_X = pca.fit_transform(scaled_X)
    if plot:
        plot_pca_exp_var(pca)
    n_pcs = next(i for i, val in enumerate(np.cumsum(pca.explained_variance_ratio_)) if val > 0.95) + 1
    print(n_pcs)

    model = LinearRegression().fit(red_X[:, :n_pcs], y)
    return (model, pca, scaler)

def plot_pca_exp_var(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.axhline(y = 0.95, color = 'r', ls = '--')
    plt.axhline(y = 0.99, color = 'k', ls = '--')
    plt.xlabel('Cumulative Explained Variance')
    plt.ylabel('Number of PC')
    plt.show()

def plot_coef_path(lambdas, coefs, regr_name, best_lambda = None):
    lambdas = np.log10(lambdas)
    plt.figure()
    for i in range(coefs.shape[0]):
        l1 = plt.plot(lambdas, coefs[i,:])
    
    if best_lambda:
        l2 = plt.axvline(x = -np.log10(best_lambda), color = 'k', ls = '--')
    
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficients")
    plt.title("{} Coefficient Path".format(regr_name))
    if(best_lambda):
        plt.legend((l1[-1], l2), (regr_name, "Optimal Lambda"), loc = "upper left")
    plt.show()

def test_error(model, test_data, type = 'std', scaler = None, pca_trans = None):
    if type == 'std':
        features = model.model.exog_names
        predictions = model.predict(test_data.loc[:,features])
    if type == 'scaled':
        scaled_test = scaler.transform(test_data)
        predictions = model.predict(scaled_test[:,1:])
    if type == 'pca':
        n_pcs = model.n_features_in_
        scaled_test = scaler.transform(test_data)
        pca_test = pca_trans.transform(scaled_test[:,1:])[:,:n_pcs]
        predictions = model.predict(pca_test)

    test_error = np.sum((test_data.iloc[:,0] - predictions) ** 2) / predictions.size
    return test_error

def get_aic(X, y):
    model = sm.OLS(y, X).fit()
    return model.aic

if __name__ == "__main__":
    main()