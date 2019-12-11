def gradient_descent(input=2, target=0.8, learning_rate=0.1,
                     weight=0.5, nb_epochs=50):

    for _ in range(nb_epochs):
        pred = input * weight
        error = (pred - target)**2
        derivative = input * (pred - target)
        weight -= learning_rate * derivative

    print('Epochs: ', nb_epochs,
          'error: ', error,
          'prediction: ', pred,
          'target: ', target)


if __name__ == '__main__':
    for epochs in [10, 20, 30, 40, 50]:
        gradient_descent(nb_epochs=epochs)
