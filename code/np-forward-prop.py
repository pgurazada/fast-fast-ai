import numpy as np


def neural_network(inputs, weights):
    '''
    return a single output using pretrained weights and an array of input 
    feature values
    '''

    return inputs.dot(weights)


def ele_mult(number, vector):
    output = np.zeros_like(vector)
    assert(len(output) == len(vector))

    for i in range(len(output)):
        output[i] = number * vector[i]

    return output


def neural_network_multi_output(inputs, weights):
    return ele_mult(input, weights)


if __name__ == '__main__':
    toes =  np.array([8.5, 9.5, 9.9, 9.0])
    wlrec = np.array([0.65, 0.8, 0.8, 0.9])
    nfans = np.array([1.2, 1.3, 0.5, 1.0])

    weights = np.array([0.1, 0.2, 0])

    print(neural_network(np.array(toes[0], wlrec[0], nfans[0]),
                         weights))

