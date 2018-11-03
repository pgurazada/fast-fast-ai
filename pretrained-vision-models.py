import numpy as np
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imread
from skimage.transform import resize

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import warnings

warnings.filterwarnings(action='ignore')

sns.set_context('talk')
sns.set_style('ticks')


def camera_grab(camera_id=0, fallback_filename=None):
    camera = cv2.VideoCapture(camera_id)
    try:
        # take 10 consecutive snapshots to let the camera automatically tune
        # itself and hope that the contrast and lightning of the last snapshot
        # is good enough.
        for _ in range(10):
            snapshot_ok, image = camera.read()
        if snapshot_ok:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print("WARNING: could not access camera")
            if fallback_filename:
                image = imread(fallback_filename)
    finally:
        camera.release()
    return image


def predict_classes(model_obj=ResNet50(weights='imagenet'),
                    image='camera-grab.jpg',
                    input_shape=(224, 224)):

    model = model_obj
    print('The input expected by the ', model.name,
          'model is : ', model.input_shape)
    
    image_resized = resize(image, input_shape,
                           preserve_range=True, mode='reflect')
    image_resized_batch = np.expand_dims(image_resized, axis=0)

    x = preprocess_input(image_resized_batch.copy())

    preds = model.predict(x)

    for _, class_name, confidence in decode_predictions(preds, top=5)[0]:
        print('  {} : {:0.3f}'.format(class_name, confidence))


if __name__ == '__main__':

    img = camera_grab(fallback_filename='696670508864045056.jpg')

    plt.imshow(img)
    plt.imsave('camera-grab.jpg', img)
    plt.show()

    predict_classes(model_obj=ResNet50(weights='imagenet'),
                    image=img,
                    input_shape=(224, 224))
    
    predict_classes(model_obj=MobileNet(weights='imagenet'),
                    image=img,
                    input_shape=(224, 224))

    predict_classes(model_obj=InceptionResNetV2(weights='imagenet',
                                                input_shape=(299, 299, 3)),
                    image=img,
                    input_shape=(299, 299))
