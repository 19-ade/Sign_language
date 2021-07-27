import cv2
import numpy as np
from models import Model, ASL_Model


def selection(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('show', image_grayscale)
    im = np.reshape(cv2.resize(image_grayscale, (28, 28)), [-1, 28, 28, 1])
    x_mean = im.mean()
    x_std = im.std()
    x_train_norm = (im - x_mean) / x_std
    y = model.prediction(x_train_norm)
    return chr(np.argmax(y) + 65)


def selection_ASL(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('see', image_grayscale)
    im = np.reshape(cv2.resize(image_grayscale, (64, 64)), [-1, 64, 64, 1])
    im = im.astype('float32') / 255.0
    y = model_ASL.prediction(im)
    y = np.argmax(y)
    #print(y)
    if y == 21:
        return 'space'
    elif y == 15:
        return 'nothing'
    elif y == 4:
        return 'del'
    elif 5 <= y <= 14:
        return chr(y + 65 - 1)
    elif 16 <= y <= 20:
        return chr(y + 65 - 2)
    elif y >= 22:
        return chr(y + 65 - 3)
    elif y <= 3:
        return chr(y + 65)


model = Model()
a, b = model.store_weights()
model.load_model(a)

model_ASL = ASL_Model()
a, c = model_ASL.store_weights()
model_ASL.load_model(a)

cam_capture = cv2.VideoCapture(0)
upper_left = (50, 50)
bottom_right = (300, 300)
while True:
    _, image_frame = cam_capture.read()
    image_frame = cv2.flip(image_frame, 1)

    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = image_frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    sketcher_rect = rect_img
    sketcher_rect_1 = selection(sketcher_rect)
    sketcher_rect_2 = selection_ASL(sketcher_rect)

    # Replacing the sketched image on Region of Interest
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image_frame,
                'MNIST:' + sketcher_rect_1,
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)
    cv2.putText(image_frame,
                'ASL:' + sketcher_rect_2,
                (200, 50),
                font, 1,
                (0, 255, 0),
                2,
                cv2.LINE_4)
    cv2.imshow("Sketcher ROI", image_frame)
    if cv2.waitKey(1) == 13:
        break

cam_capture.release()
cv2.destroyAllWindows()
