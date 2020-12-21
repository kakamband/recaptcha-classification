import os
import glob
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np


def image_resize(file, circle_crop=False):
    image_name = file.split('/')[-1].split('.')[0]
    class_name = file.split('/')[-2]
    save_path = os.path.join('data/new/eval', class_name)
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, image_name) + '.png'
    im = cv2.imread(file)
    try:
        im = cv2.resize(im, (224, 224))
        height, width, depth = im.shape
        if circle_crop:
            circle_img = np.zeros((height, width), np.uint8)
            cv2.circle(circle_img, (width // 2, height // 2), 112, 255, thickness=-1)
            masked_data = cv2.bitwise_and(im, im, mask=circle_img)
            im = np.dstack([masked_data, circle_img])
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(full_path, im, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    except Exception as e:
        print(e)


if __name__ == '__main__':
    fn = glob.glob('data/eval/**/*.*')
    with Pool(8) as p:
        p.map(partial(image_resize, circle_crop=True), fn)
