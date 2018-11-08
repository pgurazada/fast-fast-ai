def w_sum(a, b):
    assert(len(a) == len(b))
    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])
    
    return output


def neural_network_simple(input, weight=0.1):
    return input * weight


def neural_network_multi_input(input, weights):
    return w_sum(input, weights)


if __name__ == '__main__':
    # toes = current number of toes
    # wlrec = current games won (percent)
    # nfans = fan count (in millions)

    toes =  [8.5, 9.5, 9.9, 9.0]
    wlrec = [0.65, 0.8, 0.8, 0.9]
    nfans = [1.2, 1.3, 0.5, 1.0] 

    print(neural_network_multi_input([toes[0], wlrec[0], nfans[0]], 
                                     [.1, .2, 0]))
