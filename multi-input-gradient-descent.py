
def w_sum(a, b):
    assert(len(a) == len(b))
    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])

    return output


def neural_network(input, weights):
    return w_sum(input, weights)


def ele_mul(number, vector):
    output = [0, 0, 0]

    assert(len(output) == len(vector))

    for i in range(len(vector)):
        output[i] = number * vector[i]

    return output


if __name__ == '__main__':
    toes = [8.5, 9.5, 9.9, 9.0]
    wlrec = [0.65, 0.8, 0.8, 0.9]
    nfans = [1.2, 1.3, 0.5, 1.0]

    win_or_lose_binary = [1, 1, 0, 1]

    target = win_or_lose_binary[0]
    alpha = .01

    # Input corresponds to every entry
    # for the first game of the season.

    input = [toes[0], wlrec[0], nfans[0]]
    weights = [.1, .2, -.1]

    for epoch in range(3):
        pred = neural_network(input, weights)

        error = (pred - target)**2
        delta = pred - target

        weight_deltas = ele_mul(delta, input)

        print("Epoch:" + str(epoch+1))
        print("Pred:" + str(pred))
        print("Error:" + str(error))
        print("Delta:" + str(delta))
        print("Weights:" + str(weights))
        print("Weight_Deltas:")
        print(str(weight_deltas))

        for i in range(len(weights)):
            weights[i] += alpha * weight_deltas[i]
 


