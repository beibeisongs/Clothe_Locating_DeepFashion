# encoding=utf-8
# Date: 2018-10-24
# Author: MJUZY


import cv2
import random


def adjust_landmarks(shape_y, shape_x, mark_x, mark_y):
    center_x = shape_x / 2  # Note: 150.0
    center_y = shape_y / 2  # Note: 102.5

    landmark_x = int(mark_x)  # Note: 107
    landmark_y = int(mark_y)  # Note: 67

    prop_x = 224 / shape_x  # Note: Proportion for x adjusting  0.7466666666666667
    prop_y = 224 / shape_y  # Note: Proportion for y adjusting  1.0926829268292684

    new_x = int(112 + (landmark_x - center_x) * prop_x)  # Note: 79
    new_y = int(112 + (landmark_y - center_y) * prop_y)  # Note: 73

    return new_x, new_y


def show_points(x, y, im):
    cv2.circle(im, (x, y), 1, (0, 0, 255), 10)


def __data_label__(path):
    f = open(path, "r")
    lines = f.readlines()
    line_i = 0
    while line_i < len(lines):

        line_i = random.randint(0, len(lines) - 1)
        line = lines[line_i]

        step_1 = line.replace("\n", "")
        step_2 = step_1.split(" ")

        mark_i = 0
        for i in range(1, len(step_2)):
            if step_2[i - 1] == '' and step_2[i] != '':
                mark_i = i
                break

        img_Relative_path = step_2[0]
        img_RePath_parts = img_Relative_path.split('/')
        img_name = img_RePath_parts[2]
        clothe_type = img_RePath_parts[1]
        img_whole_path = "D:/Dataset_DeepFashion/img_small/" + clothe_type + '/' + img_name

        img_ori = cv2.imread(img_whole_path)
        """
        show_points(int(step_2[mark_i + 3]), int(step_2[mark_i + 4]), img_ori)
        show_points(int(step_2[mark_i + 7]), int(step_2[mark_i + 8]), img_ori)
        show_points(int(step_2[mark_i + 11]), int(step_2[mark_i + 12]), img_ori)
        show_points(int(step_2[mark_i + 15]), int(step_2[mark_i + 16]), img_ori)
        show_points(int(step_2[mark_i + 19]), int(step_2[mark_i + 20]), img_ori)
        show_points(int(step_2[mark_i + 23]), int(step_2[mark_i + 24]), img_ori)
        cv2.imshow("Image_ori", img_ori)
        """

        img = img_ori / 255
        images = cv2.resize(img, (224, 224))

        # images = load_img(img_whole_path)
        # images = img_to_array(images).astype('float32')
        image = np.expand_dims(images, axis=0)  # Note: image.shape = <class 'tuple'>: (1, 300, 300, 3)
        #   This means that you should train 1 picture per step, as the first dimension of the shape is 1
        #   But, to be exact
        #       We will train the picture in the input format of size (224, 224, 3)
        #       So we should resize the picture and the landmarks
        #
        #       Now let's start to adjust the landmarks

        new_x1, new_y1 = adjust_landmarks(img_ori.shape[0], img_ori.shape[1], step_2[mark_i + 3], step_2[mark_i + 4])
        new_x2, new_y2 = adjust_landmarks(img_ori.shape[0], img_ori.shape[1], step_2[mark_i + 7], step_2[mark_i + 8])
        new_x3, new_y3 = adjust_landmarks(img_ori.shape[0], img_ori.shape[1], step_2[mark_i + 11], step_2[mark_i + 12])
        new_x4, new_y4 = adjust_landmarks(img_ori.shape[0], img_ori.shape[1], step_2[mark_i + 15], step_2[mark_i + 16])
        new_x5, new_y5 = adjust_landmarks(img_ori.shape[0], img_ori.shape[1], step_2[mark_i + 19], step_2[mark_i + 20])
        new_x6, new_y6 = adjust_landmarks(img_ori.shape[0], img_ori.shape[1], step_2[mark_i + 23], step_2[mark_i + 24])
        """
        show_points(new_x1, new_y1, images)
        show_points(new_x2, new_y2, images)
        show_points(new_x3, new_y3, images)
        show_points(new_x4, new_y4, images)
        show_points(new_x5, new_y5, images)
        show_points(new_x6, new_y6, images)
        cv2.imshow("224^2_Images", images)
        """
        label = [new_x1 / 224, new_y1 / 224, new_x2 / 224, new_y2 / 224,
                 new_x3 / 224, new_y3 / 224, new_x4 / 224, new_y4 / 224,
                 new_x5 / 224, new_y5 / 224, new_x6 / 224, new_y6 / 224]

        # lable = [new_x1, new_y1, new_x2, new_y2,
        #          new_x3, new_y3, new_x4, new_y4,
        #          new_x5, new_y5, new_x6, new_y6]

        label_arr = np.array(label)
        label = label_arr.reshape(1, 12)

        yield (image, label)


""" Attention: 

        Sample: 
        >>>__data_label__(path='./list_landmarks.txt')
"""
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import *
import keras.optimizers as optimizers


model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)

model.layers.pop()
model.layers.pop()
model.layers.pop()

model.outputs = [model.layers[-1].output]
x = Dense(256, activation='relu')(model.layers[-1].output)
x = Dense(12)(x)

model = Model(model.input, x)
for i in range(len(model.layers)):
    if i <= 3:
        model.layers[i].trainable = False
    else:
        model.layers[i].trainable = True
# Attention: The following trying has failed
#
# for i in range(len(model.layers)):
#     if i <= len(model.layers) - 3:
#         model.layers[i].trainable = False
#     else:
#         model.layers[i].trainable = True


"""
    >>>for layer in model.layers[:10]: layer.trainable = False
"""

""" Attention:  
        loss='mse' refers to the loss function called average square loss
"""
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='mse', metrics=['acc'])
epoch_num = 100
""" Attention: The following trying has failed

model.compile(optimizer="adam", loss='mse', metrics=['acc'])
epoch_num = 100

learning_rate = np.linspace(0.03, 0.01, epoch_num)
change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
check_point = ModelCheckpoint('Clothe_Locating_12points.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                save_weights_only=False, mode='auto', period=1)
"""

model.summary()

model.fit_generator(__data_label__(path='./list_landmarks.txt'), callbacks=None,
                    samples_per_epoch=100,
                    epochs=epoch_num,
                    validation_steps=30,
                    validation_data=__data_label__(path='./list_landmarks.txt'))

model.save("Clothe_Locating_12points.h5")
