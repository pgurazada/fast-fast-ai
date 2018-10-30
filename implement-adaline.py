import numpy as np
import pandas as pd


class AdalineGD:

    def __init__(self, eta=0.01, n_epochs=50):
        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_epochs):
            output = self.net_input(X)
            errors = (y - output)

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)

        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    print('The shape of X: ', X.shape)

    model_adaline_large_lr = AdalineGD(eta=0.01, n_epochs=10)
    model_adaline_large_lr.fit(X, y)

    model_adaline_small_lr = AdalineGD(eta=0.0001, n_epochs=10)
    model_adaline_small_lr.fit(X, y)

    print(model_adaline_large_lr.cost_)
    print(model_adaline_small_lr.cost_)

