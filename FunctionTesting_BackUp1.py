# encoding=utf-8
# Date: 2018-10-24
# Author: MJUZY


import cv2
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.transform as transform


def __data_label__(path):
    f = open(path, "r")
    line_i = 0
    for line in f.readlines():

        if 2 <= line_i <= 30000:  # Note:
            #   'img/Sheer_Pleated-Front_Blouse/img_00000001.jpg                         1  0 146 102  0 173 095  0 094 242  0 205 255  0 136 229  0 177 232 '

            step_1 = line.replace("\n", "")
            step_2 = step_1.split(" ")

            img_Relative_path = step_2[0]
            img_RePath_parts = img_Relative_path.split('/')
            img_name = img_RePath_parts[2]
            clothe_type = img_RePath_parts[1]
            img_whole_path = "D:/Dataset：DeepFashion/img_small/" + clothe_type + '/' + img_name

            images = load_img(img_whole_path)
            images = img_to_array(images).astype('float32')
            image = np.expand_dims(images, axis=0)

            lable = [int(step_2[28]), int(step_2[29]), int(step_2[32]), int(step_2[33]), int(step_2[36]),
                     int(step_2[37]), int(step_2[40]), int(step_2[41]), int(step_2[44]), int(step_2[45]),
                     int(step_2[48]), int(step_2[49])]
            lables = np.array(lable)
            lables = lables.reshape(1, 12)

            yield (image, lables)

        line_i += 1


def show_points(x, y, img):
    cv2.circle(img, (x, y), 1, (0, 0, 255), 10)


def adjust_landmarks(shape_y, shape_x, mark_x, mark_y):
    center_x = shape_x / 2  # Note: 150.0
    center_y = shape_y / 2  # Note: 102.5

    landmark_x = int(mark_x)  # Note: 107
    landmark_y = int(mark_y)  # Note: 67

    prop_x = 224 / shape_x  # Note: Proportion for x adjusting  0.7466666666666667
    prop_y = 224 / shape_y  # Note: Proportion for y adjusting  1.0926829268292684

    new_x = int(112 + (landmark_x - center_x) * prop_x)     # Note: 79
    new_y = int(112 + (landmark_y - center_y) * prop_y)     # Note: 73

    return new_x, new_y


path = './list_landmarks.txt'
f = open(path, "r")
line_i = 0
for line in f.readlines():

    if 2 <= line_i <= 30000:    # Note:
                                #   'img/Sheer_Pleated-Front_Blouse/img_00000001.jpg                         1  0 146 102  0 173 095  0 094 242  0 205 255  0 136 229  0 177 232 '

        step_1 = line.replace("\n", "")
        step_2 = step_1.split(" ")

        img_Relative_path = step_2[0]
        img_RePath_parts = img_Relative_path.split('/')
        img_name = img_RePath_parts[2]
        clothe_type = img_RePath_parts[1]
        img_whole_path = "D:/Dataset_DeepFashion/img_small/" + clothe_type + '/' + img_name

        img = cv2.imread(img_whole_path)

        images = load_img(img_whole_path)
        images = img_to_array(images).astype('float32')
        image = np.expand_dims(images, axis=0)  # Note: image.shape = <class 'tuple'>: (1, 300, 300, 3)
        #   This means that you should train 1 picture per step, as the first dimension of the shape is 1
        #   But, to be exact
        #       We will train the picture in the input format of size (224, 224, 3)
        #       So we should resize the picture and the landmarks
        #
        #       Now let's start to adjust the landmarks
        new_x1, new_y1 = adjust_landmarks(images.shape[0], images.shape[1], step_2[28], step_2[29])
        new_x2, new_y2 = adjust_landmarks(images.shape[0], images.shape[1], step_2[32], step_2[33])
        new_x3, new_y3 = adjust_landmarks(images.shape[0], images.shape[1], step_2[36], step_2[37])
        new_x4, new_y4 = adjust_landmarks(images.shape[0], images.shape[1], step_2[40], step_2[41])
        new_x5, new_y5 = adjust_landmarks(images.shape[0], images.shape[1], step_2[44], step_2[45])
        new_x6, new_y6 = adjust_landmarks(images.shape[0], images.shape[1], step_2[48], step_2[49])
        shrink = cv2.resize(img, (224, 224))    # Note: ', interpolation=cv2.INTER_AREA'
        # show_points(new_x1, new_y1, shrink)
        # show_points(new_x2, new_y2, shrink)
        # show_points(new_x3, new_y3, shrink)
        # show_points(new_x4, new_y4, shrink)
        show_points(new_x5, new_y5, shrink)
        show_points(new_x6, new_y6, shrink)
        cv2.imshow("Shrink", shrink)

        # img_io = io.imread(img_whole_path)
        # img_tr = transform.resize(img_io, (224, 224))
        # plt.figure("resize")
        # plt.title("resize")
        # plt.imshow(img_tr)
        # plt.show()

        lable = [int(step_2[28]), int(step_2[29]), int(step_2[32]), int(step_2[33]), int(step_2[36]),
                 int(step_2[37]), int(step_2[40]), int(step_2[41]), int(step_2[44]), int(step_2[45]),
                 int(step_2[48]), int(step_2[49])]

        show_points(int(step_2[28]), int(step_2[29]), img)
        show_points(int(step_2[32]), int(step_2[33]), img)
        show_points(int(step_2[36]), int(step_2[37]), img)
        show_points(int(step_2[40]), int(step_2[41]), img)
        show_points(int(step_2[44]), int(step_2[45]), img)
        show_points(int(step_2[48]), int(step_2[49]), img)

        cv2.imshow("Image", img)

    line_i += 1

__data_label__(path='./list_landmarks.txt')
