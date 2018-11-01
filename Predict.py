# encoding=utf-8
# Date: 2018-10-27
# Author: MJUZY


import cv2
from keras.models import load_model
import numpy as np


def show_points(x, y, im):
    cv2.circle(im, (x, y), 1, (0, 0, 255), 10)


def showPrediction(img_path):

        img_ori = cv2.imread(img_path)

        img = img_ori / 255
        images = cv2.resize(img, (224, 224))
        image = np.expand_dims(images, axis=0)  # Note: image.shape = <class 'tuple'>: (1, 300, 300, 3)

        result = model.predict(image)
        content = result[0]
        content = content * 224

        show_points(int(content[0]), int(content[1]), images)
        show_points(int(content[2]), int(content[3]), images)
        show_points(int(content[4]), int(content[5]), images)
        show_points(int(content[6]), int(content[7]), images)
        show_points(int(content[8]), int(content[9]), images)
        show_points(int(content[10]), int(content[11]), images)
        cv2.imshow("224^2_Images", images)
        cv2.waitKey(0)


model = load_model('Clothe_Locating_12points.h5')
model.summary()

img_path = "D:/Dataset_DeepFashion/img_small/" + "Sheer_Woven_Blouse/img_00000101.jpg"
showPrediction(img_path)
