import cv2
import numpy as np
from models import Model, ASL_Model

def empty(a):
    pass

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

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 39, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 126, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 194, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 110, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

while True:
    _, img = cam_capture.read()
    img = cv2.flip(img, 1)
    ###########################
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.bitwise_not(cv2.inRange(imgHsv, lower, upper))
    image_frame = cv2.bitwise_and(img, img, mask=mask)
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
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, image_frame])
    cv2.imshow('Horizontal Stacking', hStack)

    if cv2.waitKey(1) == 13:
        break

cam_capture.release()
cv2.destroyAllWindows()
