import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import Model
def input():
    label_train, pixel_train = pd.read_csv("sign_mnist_train.csv").iloc[:, 0], pd.read_csv("sign_mnist_train.csv").iloc[:, 1:]
    label_test, pixel_test = pd.read_csv("sign_mnist_test.csv").iloc[:, 0], pd.read_csv("sign_mnist_test.csv").iloc[:, 1:]
    print(pixel_train.shape)
    x_train = pixel_train.values
    x_test = pixel_test.values
    Y_test = label_test.values
    Y_train = label_train.values
    x_train_reshaped = x_train.reshape([-1, 28, 28, 1])
    x_test_reshaped = x_test.reshape([-1, 28, 28, 1])


def plot_data(x_train_reshaped, Y_train):
    rows = 5 # defining no. of rows in figure
    cols = 6 # defining no. of colums in figure

    f = plt.figure(figsize=(2*cols, 2*rows))  # defining a figure

    for i in range(rows*cols):
        f.add_subplot(rows, cols, i+1) # adding sub plot to figure on each iteration
        plt.imshow(x_train_reshaped[i].reshape([28, 28]))
        plt.axis("off")
        plt.title(chr(65 + Y_train[i]), y=-0.15, color="green")
    plt.savefig("digits.png")


def preprocessing(x_train_reshaped, x_test_reshaped):
    x_mean = x_train_reshaped.mean()
    x_std = x_train_reshaped.std()
    x_test_norm = (x_test_reshaped-x_mean)/x_std
    x_train_norm = (x_train_reshaped-x_mean)/x_std


def model_training_eval(x_train_norm, x_test_norm, Y_test, Y_train):
    model = Model()
    model.describe()
    checkp, cp_callback = model.store_weights()
    history = model.fit_model(x_train_norm,Y_train, cp_callback)

    model.plots_acc(history)
    model.plots_loss(history)
    score = model.score(x_test_norm, Y_test)
    print('Test accuarcy: %0.2f%%' % (score[1] * 100))


def load_evaluate(x_test_norm, Y_test):
    model = Model()
    model.describe()
    checkp, cp_callback = model.store_weights()
    model.load_model(checkp)
    score = model.score(x_test_norm, Y_test)
    print('Test accuarcy: %0.2f%%' % (score[1] * 100))
