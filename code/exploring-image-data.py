import numpy as np

import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imread

import warnings

sns.set_context('talk')
sns.set_style('ticks')

warnings.filterwarnings(action='ignore')


def camera_grab(camera_id=0, fallback_filename=None):
    camera = cv2.VideoCapture(camera_id)
    try:
        # take 10 consecutive snapshots to let the camera automatically tune
        # itself and hope that the contrast and lightning of the last snapshot
        # is good enough.
        for i in range(10):
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

# Let us first read in a jpeg image as an array and understand how the image is 
# stored and interpreted as a numpy array.

image = imread('696670508864045056.jpg')

print(type(image))
print(image.shape)
print(image.dtype)
print(image.min(), image.max())

red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

grey_image = image.mean(axis=2)

#print(red_channel.min(), red_channel.max())

plt.figure(figsize=(12, 12))

plt.subplot(511)
plt.imshow(image)

plt.subplot(512)
plt.imshow(red_channel, cmap=plt.cm.Reds_r)

plt.subplot(513)
plt.imshow(green_channel, cmap=plt.cm.Greens_r)

plt.subplot(514)
plt.imshow(blue_channel, cmap=plt.cm.Blues_r)

plt.subplot(515)
plt.imshow(grey_image, cmap=plt.cm.Greys_r)

plt.tight_layout()

plt.show()

webcam_image = camera_grab(camera_id=0, fallback_filename='696670508864045056.jpg')

plt.imshow(webcam_image)
plt.show()
