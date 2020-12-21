import numpy as np
import cv2


def visualize_numpy_ds(img: np.ndarray, label: np.ndarray, classes: list):
    if len(img.shape) == 4:
        img = img[0, :, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('im', img)
    cv2.setWindowTitle('im', classes[np.argmax(label).squeeze()])
    key = cv2.waitKey(0)
    if key == ord('q'):
        return 0
    if key == ord('n'):
        return 1
