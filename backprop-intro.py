import numpy as np


def relu(x):
    return (x > 0) * x


def deriv_relu(output):
    return (output > 0)


def basic_neural_net(data, targets, weights, learning_rate=.1, nb_epochs=50):

    for _ in range(nb_epochs):

        error_cumulative = 0

        for row_index in range(len(targets)):
            input = data[row_index]
            target = targets[row_index]
            prediction = input.dot(weights)

            error = (prediction - target)**2
            error_cumulative += error

            delta = prediction - target
            weights -= (learning_rate * input * delta)
            print('prediction: ', str(prediction))

        print('Error: ', str(error_cumulative))


def one_backprop_iteration(data, targets,
                           learning_rate=0.2,
                           hidden_layer_units=3):

    weights_0_1 = 2 * np.random.random((3, hidden_layer_units)) - 1
    weights_1_2 = 2 * np.random.random((hidden_layer_units, 1)) - 1

    # Forward pass
    layer_0 = data[0:1]
    layer_1 = np.dot(layer_0, weights_0_1)
    layer_1 = relu(layer_1)
    layer_2 = np.dot(layer_1, weights_1_2)

    error = (layer_2 - targets[0:1])**2

    # Backward pass

    layer_2_delta = (layer_2 - targets[0:1])
    layer_1_delta = layer_2_delta.dot(weights_1_2.transpose())
    layer_1_delta *= deriv_relu(layer_1)

    weight_delta_1_2 = layer_1.transpose().dot(layer_2_delta)
    weight_delta_0_1 = layer_0.transpose().dot(layer_1_delta)

    weights_1_2 -= learning_rate * weight_delta_1_2
    weights_0_1 -= learning_rate * weight_delta_0_1

    print('Error: ', error)
    print('Weights of hidden layer: ', weights_0_1)
    print('Weghts of output layer', weights_1_2)


if __name__ == '__main__':

    streetlights = np.array([[1, 0, 1],
                             [0, 1, 1],
                             [0, 0, 1],
                             [1, 1, 1],
                             [0, 1, 1],
                             [1, 0, 1]])

    lights = np.array([[1, 0, 1],
                       [0, 1, 1],
                       [0, 0, 1],
                       [1, 1, 1]])

    walk_or_stop = np.array([0, 1, 0, 1, 1, 0])
    walk_stop = np.array([[1, 1, 0, 0]]).transpose()

    weights = np.array([.50, .48, -.70])

    basic_neural_net(data=streetlights,
                     targets=walk_or_stop,
                     weights=weights)

    one_backprop_iteration(data=lights,
                           targets=walk_stop)
