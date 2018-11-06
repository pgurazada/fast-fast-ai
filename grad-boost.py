import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression


def loss(y, y_hat): 
    return (.5 * (y - y_hat)**2).mean()


def grad_loss(y, y_hat): 
    return (y - y_hat)


def grad_boost(X, y, grad_fn, loss_fn,
               model_obj=LinearRegression(),
               learning_rate=0.01, nb_epochs=50, y_hat_init=0):

    fit = y_hat_init

    u = grad_fn(y, fit)

    theta = np.zeros(X.shape[1])

    loss = []

    for _ in range(nb_epochs):

        model_obj.fit(X, u)

        theta_i = model_obj.coef_

        theta += learning_rate * theta_i
        fit += learning_rate * model_obj.predict(X)

        u = grad_fn(y, fit)

        loss.append(loss_fn(y, fit))

    return loss, theta, nb_epochs


if __name__ == '__main__':
    X, y, coef = make_regression(n_samples=1000, 
                                 n_features=4, 
                                 noise=0.1, 
                                 coef=True)

    loss, theta, nb_epochs = grad_boost(X, y, grad_loss, loss, nb_epochs=50)

    print('Actual coefficients: ', coef)
    print('Coefficients from gradient boosted model after',  nb_epochs,
          ' epochs: ', theta)
