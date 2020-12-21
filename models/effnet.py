from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, Dropout


def get_captcha_model():
    backbone = EfficientNetB4(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet',
        pooling='avg',
    )
    backbone.trainable = False

    m = Sequential([
        backbone,
        Dropout(0.4),
        Dense(23, activation='softmax')
    ])

    return m


if __name__ == '__main__':
    captcha_model = get_captcha_model()
    captcha_model.summary()
