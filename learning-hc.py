def learn_hl(input=0.5, target=0.8, nb_epochs=1000, weight_step=.001):

    weight = 0.5

    for _ in range(nb_epochs):
        
        prediction = input * weight
        error = (prediction - target)**2

        #print('Error:' + str(error), ' Prediction: ' + str(prediction))

        # To step up or down, that is the question

        up_prediction = input * (weight + weight_step)
        up_error = (up_prediction - target)**2

        down_prediction = input * (weight - weight_step)
        down_error = (down_prediction - target)**2

        if up_error > down_error:
            weight -= weight_step
        else:
            weight += weight_step

    print('Epochs: ', nb_epochs,
          'Final prediction: ', input*weight,
          'Target: ', target)


if __name__ == '__main__':
    for epochs in [500, 1000, 2000]:
        learn_hl(nb_epochs=epochs)






