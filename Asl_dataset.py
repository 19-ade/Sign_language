import os
import cv2
import numpy as np
import pandas as pd
from models import ASL_Model
import random

DATADIR = 'archive(1)/asl_alphabet_train/asl_alphabet_train'
CATEGORIES = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q',
              'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
test_dir = 'archive(1)/asl_alphabet_test/asl_alphabet_test'


def get_data_training(path_dir, df):
    df = []
    for i in range(len(CATEGORIES)):
        path = os.path.join(path_dir, CATEGORIES[i])
        category = i
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array = cv2.resize(img_array, (64, 64))
            df.append([img_array, category])
    random.shuffle(df)
    return df


def pre_processing(df):
    X = []
    y = []
    for features, label in df:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape([-1, 64, 64, 1])
    X = X.astype('float32') / 255.0
    y = np.array(y)
    return X, y


def testing_data(df, path):
    df = []

    for image in os.listdir(path):
        name = 0
        img = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
        final_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final_img = cv2.resize(final_img, (64, 64))
        if image != 'space.jpg' and image != 'nothing.jpg' and image != 'del.jpg':
            names = (ord(image.split('.')[0]) - 65)
        else:
            if image == 'space.jpg':
                names = 23
            elif image == 'del.jpg':
                names = 4
            else:
                names = 15
        df.append([final_img, names])
    random.shuffle(df)
    return df


df = []
df = get_data_training(DATADIR, df)
X_train, y_train = pre_processing(df)
# print(X_train.shape)
print(y_train.shape)
# print(y_train[0])
# df_test = []
# df_test = testing_data(df_test, test_dir)
# X_test, y_test = pre_processing(df_test)
# print(X_test.shape)
model = ASL_Model()
# model.describe()
checkp, cp_callback = model.store_weights()
history = model.fit_model(X_train, y_train, cp_callback)

model.plots_acc(history)
model.plots_loss(history)

# model.load_model(checkp)
# score = model.score(X_test, y_test)
# print('Test accuarcy: %0.2f%%' % (score[1] * 100))
