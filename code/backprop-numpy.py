import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Series of helper functions


def to_categorical(n_classes, y):
    return np.eye(n_classes)[y]


def sigmoid(X):
    return 1/(1 + np.exp(X))


def dsigmoid(X):
    return sigmoid(X) * (1 - sigmoid(X))


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)


def cross_entropy(y_true, y_pred, EPSILON=1e-8):
    y_true, y_pred = np.atleast_2d(y_true), np.atleast_2d(y_pred)
    logloss = np.sum(np.log(EPSILON + y_pred) * y_true, axis=1)
    return -np.mean(logloss)


class LogisticRegression:
    
    def __init__(self, input_size, output_size):
        self.W = np.random.uniform(high=.1, low=-.1, 
                                   size=(input_size, output_size))
        self.b = np.random.uniform(low=-.1, high=.1, size=output_size)
        self.output_size = output_size

    def forward(self, X):
        Z = np.dot(X, self.W) + self.b
        return softmax(Z)

    def predict(self, X):
        if len(X.shape) == 1:
            return np.argmax(self.forward(X))
        else:
            return np.argmax(self.forward(X), axis=1)

    def grad_loss(self, X, y_true):
        y_pred = self.forward(X)
        loss_out = y_pred - to_categorical(self.output_size, y_true)
        grad_W = np.outer(X, loss_out)
        grad_b = loss_out

        return {'W': grad_W, 'b': grad_b} 

    def train(self, X, y, learning_rate):
        grads = self.grad_loss(X, y)
        self.W -= learning_rate * grads['W']
        self.b -= learning_rate * grads['b']

    def loss(self, X, y):
        return cross_entropy(to_categorical(self.output_size, y), 
                             self.forward(X))

    def accuracy(self, X, y):
        y_preds = np.argmax(self.forward(X), axis=1)
        return np.mean(y_preds == y)


class NeuralNet:
    '''
    single hidden layer or arbitrary size

    '''
    def __init__(self, input_size, hidden_size, output_size):
        self.W_h = np.random.uniform(low=-.01, high=.01, 
                                     size=(input_size, hidden_size))
        self.b_h = np.zeros(hidden_size)
        self.W_o = np.random.uniform(low=-.01, high=.01,
                                     size=(hidden_size, output_size))
        self.b_o = np.zeros(output_size)
        self.output_size = output_size

    def forward(self, X):
        h = sigmoid(np.dot(X, self.W_h) + self.b_h)
        y = softmax(np.dot(h, self.W_o) + self.b_o)
        return y
       
    def forward_keep_activations(self, X):
        z_h = np.dot(X, self.W_h) + self.b_h
        h = sigmoid(z_h)
        z_o = np.dot(h, self.W_o) + self.b_o
        y = softmax(z_o)

        return y, h, z_h

    def loss(self, X, y):
        return cross_entropy(to_categorical(self.output_size, y),
                             self.forward(X))

    def grad_loss(self, X, y_true):
        y, h, z_h = self.forward_keep_activations(X)
        grad_z_o = y - to_categorical(self.output_size, y_true)

        grad_W_o = np.outer(h, grad_z_o)
        grad_b_o = grad_z_o

        grad_h = np.dot(grad_z_o, np.transpose(self.W_o))
        grad_z_h = grad_h * dsigmoid(z_h)

        grad_W_h = np.outer(X, grad_z_h)
        grad_b_h = grad_z_h

        grads = {'W_h': grad_W_h, 'b_h': grad_b_h,
                 'W_o': grad_W_o, 'b_o': grad_b_o}
        
        return grads

    def train(self, X, y, learning_rate):
        grads = self.grad_loss(X, y)
        self.W_h -= learning_rate * grads['W_h']
        self.b_h -= learning_rate * grads['b_h']
        self.W_o -= learning_rate * grads['W_o'] 
        self.b_o -= learning_rate * grads['b_o']

    def predict(self, X):
        if len(X.shape) == 1:
            return np.argmax(self.forward(X))
        else:
            np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        y_preds = np.argmax(self.forward(X), axis=1)
        return np.mean(y_preds == y)


if __name__ == '__main__':

    # data

    digits = load_digits()

    data = np.asarray(digits.data, dtype='float32')
    target = np.asarray(digits.target, dtype='int32')

    X_train, X_test, y_train, y_test = train_test_split(data, target, 
                                                        test_size=0.2, 
                                                        random_state=20180810)               
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    model_logit = LogisticRegression(n_features, n_classes)

    print('Evaluation of the untrained model:')
    print('------------------------------')

    train_loss = model_logit.loss(X_train, y_train)
    train_acc = model_logit.accuracy(X_train, y_train)
    test_acc = model_logit.accuracy(X_test, y_test)

    print('Train loss: %.3f, train accuracy: %.3f, test accuracy: %.3f' 
          % (train_loss, train_acc, test_acc))

    # Testing the neural net

    n_hidden = 10

    model = NeuralNet(n_features, n_hidden, n_classes)

    losses, accuracies, accuracies_test = [], [], []

    losses.append(model.loss(X_train, y_train))
    accuracies.append(model.accuracy(X_train, y_train))
    accuracies_test.append(model.accuracy(X_test, y_test))

    print("Random init: train loss: %0.5f, train acc: %0.3f, test acc: %0.3f"
          % (losses[-1], accuracies[-1], accuracies_test[-1]))
    
    for epoch in range(10):
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            model.train(x, y, 0.001)

        losses.append(model.loss(X_train, y_train))
        accuracies.append(model.accuracy(X_train, y_train))
        accuracies_test.append(model.accuracy(X_test, y_test))
        print("Epoch #%d, train loss: %0.5f, train acc: %0.3f, test acc: %0.3f"
              % (epoch + 1, losses[-1], accuracies[-1], accuracies_test[-1]))

    plt.plot(losses)
    plt.show()



    




