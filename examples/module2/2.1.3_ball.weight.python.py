import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def main():
    pass

def matrix_method():
    y = np.array([[2,3,4]]).T
    A = np.array([[1, 0, 1]]).T
    B = np.array([[0, 1, 1]]).T
    X = np.concatenate([A, B], axis = 1)

    XTX = np.dot(X.T, X)
    sigma = np.sqrt(1/np.linalg.det(XTX))
    XTX_inv = np.linalg.inv(XTX)
    beta = np.dot(XTX_inv, X.T)
    beta = np.dot(beta, y)
    print(beta)
    print(sigma)

    t_70 = 1.963
    int_width = t_70 * sigma * np.sqrt(np.diag(XTX_inv)[0])
    print("[{}, {}]".format(beta[0,0] - int_width, beta[0,0] + int_width))

    x_new = np.array([[1,1]]).T
    tmp = np.dot(x_new.T, XTX_inv)
    tmp = np.dot(tmp, x_new)
    tmp = tmp[0,0]
    a_plus_b = beta[0,0] + beta[1,0]
    pred_width = t_70 * sigma * np.sqrt(1 + tmp)
    print("[{}, {}]".format(a_plus_b - pred_width, a_plus_b + pred_width))

def slr():
    y = np.array([[2,3,4]]).T
    A = np.array([[1, 0, 1]]).T
    B = np.array([[0, 1, 1]]).T
    X = np.concatenate([A, B], axis = 1)

    model = sm.OLS(y, X)
    results = model.fit()
    pred = results.get_prediction(np.array([1,1]))
    print(results.summary())
    print(results.conf_int(alpha = 0.3))
    print(pred.summary_frame(alpha = 0.3))


if __name__ == "__main__":
    matrix_method()