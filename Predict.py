# encoding=utf-8
# Date: 2018-10-27
# Author: MJUZY

import cv2
from keras.models import load_model


def show_points(x, y, img):
    cv2.circle(img, (x, y), 1, (0, 0, 255), 10)


model = load_model('Clothe_Locating_12points.h5')
model.summary()

from keras.preprocessing import image
import numpy as np

img_path = "D:/Dataset_DeepFashion/img_small/" + "Sheer_Pleated-Front_Blouse/img_00000111.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)  # shape = <class 'tuple'>: (224, 224, 3)
""" Description : of np.expand_dims

    For example, originally shape = (2,3), and axis=0
        then shape changed into (1,2,3)
            when axis = 1then shape changed into (2,1,3)
"""
img_tensor = np.expand_dims(img_tensor, axis=0)  # shape = <class 'tuple'>: (1, 224, 224, 3)
img_tensor /= 255.
print(img_tensor.shape)  # output: (1, 224, 224, 3)

result = model.predict(img_tensor)
content = result[0]
content = content * 224

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))

show_points(int(content[0]), int(content[1]), img)
show_points(int(content[2]), int(content[3]), img)
show_points(int(content[4]), int(content[5]), img)
show_points(int(content[6]), int(content[7]), img)
show_points(int(content[8]), int(content[9]), img)
show_points(int(content[10]), int(content[11]), img)

cv2.imshow("Image", img)
cv2.waitKey()
cv2.destroyAllWindows()