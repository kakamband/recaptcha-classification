import tensorflow as tf

from data.dataset import Dataset, get_label
from models.effnet import get_captcha_model


train_path = 'data/train'
test_path = 'data/eval'

classes = get_label(train_path, test_path)
train_ds = Dataset(train_path, labels=classes, batch_size=32, is_training=True, random_seed=20).ds
test_ds = Dataset(test_path, labels=classes, batch_size=32, is_training=False, random_seed=20).ds

model = get_captcha_model()
opt = tf.keras.optimizers.Adam(lr=0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=3)]

model.compile(opt, loss, metrics)
model.fit(train_ds, validation_data=test_ds, epochs=20)
