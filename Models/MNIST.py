import numpy as np
import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(
    train_images.shape[0], train_images.shape[1], train_images.shape[2], 1
)
images_test = test_images.reshape(
    test_images.shape[0], test_images.shape[1], test_images.shape[2], 1
)

train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

input_tensor_shape = (test_images.shape[1], train_images.shape[2], 1)

print(input_tensor_shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_tensor_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), padding="valid"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(train_images, train_labels, epochs=5, batch_size=500)
model.summary()
res = model.evaluate(test_images, test_labels)
print("Model1 has an accuracy of {0:.2f}%".format(res[1] * 100))

converter = tf.lite.TFLiteConverter.from_keras_model(model)

globals()["data"] = train_images


def representative_dataset():
    for i in range(500):
        yield [np.array(globals()["data"][i : i + 1])]


print(np.array(train_images[0]).shape)
print(np.array(train_images[0]))

converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_new_quantizer = True

tflite_quant_model = converter.convert()

with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_quant_model)
