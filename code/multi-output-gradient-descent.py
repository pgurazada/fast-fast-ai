def ele_mul(scalar, vector):
    out = [0, 0, 0]

    for i in range(len(out)):
        out[i] = vector[i] * scalar

    return out


def neural_network(input, weights):
    return ele_mul(input, weights)


if __name__ == '__main__':
    weights = [.3, .2, .9]
    alpha = .1

    wlrec = [0.65, 1.0, 1.0, 0.9]

    hurt = [.1, .0, .0, .1]
    win = [1, 1, 0, 1]
    sad = [.1, .0, .1, .2]

    input = wlrec[0]
    target = [hurt[0], win[0], sad[0]]

    pred = neural_network(input, weights)
    
    error = [0, 0, 0]
    delta = [0, 0, 0]

    for i in range(len(error)):
        error[i] = (pred[i] - target[i])**2
        delta[i] = pred[i] - target[i]

    weight_deltas = ele_mul(input, delta)

    for i in range(len(weights)):
        weights[i] += (weight_deltas[i] * alpha)
    
    print('weights: ', weights)
    print('weights deltas: ', weight_deltas)




