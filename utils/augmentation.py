import tensorflow as tf
import cv2
import numpy as np
import albumentations as A


transformations = {
    'train_transform': A.Compose([
        A.Rotate(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA,  always_apply=True),
        A.Flip(),
        A.OneOf([
            A.CLAHE(tile_grid_size=(5, 5)),
            A.RandomBrightnessContrast(),
            A.RandomGamma()
        ]),
        A.OneOf([
            A.RGBShift(),
            A.HueSaturationValue(),
            A.ChannelShuffle(p=0.25)
        ]),
        A.Downscale(scale_min=0.3, scale_max=0.5, always_apply=True),
        A.GaussNoise(var_limit=(20., 90.), always_apply=True),
        A.Blur(blur_limit=12, always_apply=True),
        A.Resize(224, 224, always_apply=True),
        A.ToFloat()
    ]),
    'val_transform': A.Compose([
        A.Resize(224, 224, always_apply=True),
        A.ToFloat()
    ])
}


def apply_augmentation(image, is_training):
    if is_training:
        data = transformations['train_transform'](image=image)
    else:
        data = transformations['val_transform'](image=image)
    image = data['image']
    return image


def aug_func(image, label, is_training=False):
    image_transformed = tf.numpy_function(apply_augmentation, inp=[image, is_training], Tout=tf.float32, name='aug_func')
    image_transformed = tf.multiply(image_transformed, 255)
    return image_transformed, label


def debug_augmentation(trans_func, image):
    data = trans_func(image=image[0])
    cv2.imshow('train_normal', image[0])
    cv2.imshow('test_normal', image[1])
    while True:
        cv2.imshow('augmented', data['image'])
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('n'):
            data = trans_func(image=image[0])
            cv2.imshow('augmented', data['image'])
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train_image = cv2.imread('../data/train/bus/bus_001.png', 1)
    test_image = cv2.imread('../data/eval/bus/bus_001.png', 1)
    debug_augmentation(trans_func=transformations['train_transform'], image=[train_image, test_image])
