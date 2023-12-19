import numpy as np
import tensorflow as tf
from tensorflow import keras

import pathlib, os
import pickle
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b"label_names"]
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b"data"]
        else:
            cifar_train_data = np.vstack(
                (cifar_train_data, cifar_train_data_dict[b"data"])
            )
        cifar_train_filenames += cifar_train_data_dict[b"filenames"]
        cifar_train_labels += cifar_train_data_dict[b"labels"]

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b"data"]
    cifar_test_filenames = cifar_test_data_dict[b"filenames"]
    cifar_test_labels = cifar_test_data_dict[b"labels"]

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return (
        cifar_train_data,
        cifar_train_filenames,
        to_categorical(cifar_train_labels),
        cifar_test_data,
        cifar_test_filenames,
        to_categorical(cifar_test_labels),
        cifar_label_names,
    )

tmp_dir = "/tmp"
zip_dir = tmp_dir + "/zips"

def GetData(dataset_dir):

    import wget, os

    global tmp_dir
    global zip_dir
    dataset_name = "Cifar10"
    dataset_zip = "cifar-10-python.tar.gz"
    dataset_url = "https://www.cs.toronto.edu/~kriz/{}".format(dataset_zip)

    if dataset_dir != "":
        tmp_dir = dataset_dir
        zip_dir = dataset_dir + "/zips"

    make_dataset_dirs(dataset_name)

    if not os.path.isfile(os.path.join(zip_dir, dataset_zip)):
        print("Downloading Cifar10 dataset tar into: {}".format(zip_dir))
        wget.download(dataset_url, out=zip_dir, bar=bar_progress)
    else:
        print("Cifar10 tar already exists, skipping download")

    output_dir = os.path.join(tmp_dir, dataset_name)

    if not len(os.listdir(output_dir)):
        import tarfile

        tar_filepath = "{}/{}".format(zip_dir, dataset_zip)
        print("Extracting Cifar10 tar to: {}".format(tar_filepath))
        tar = tarfile.open(tar_filepath, "r:gz")
        tar.extractall(path=output_dir)
        tar.close()

        import shutil

        out_files = os.listdir(output_dir)

        sub_out_files = os.listdir(os.path.join(output_dir, out_files[0]))

        print(
            "Moving files {} into parent directory {}".format(sub_out_files, output_dir)
        )

        for file in sub_out_files:
            shutil.move(os.path.join(output_dir, out_files[0], file), output_dir)

        shutil.rmtree(os.path.join(output_dir, out_files[0]))
    else:
        print("Cifar10 tar already extracted")

    (
        train_data,
        train_filenames,
        train_labels,
        test_data,
        test_filenames,
        test_labels,
        label_names,
    ) = load_cifar_10_data(output_dir)

    input_shape = GetInputShape()

    return {
        "test_data": test_data,
        "train_data": train_data,
        "test_labels": test_labels,
        "train_labels": train_labels,
        "input_tensor_shape": input_shape,
    }


def GetInputShape():
    return (32, 32, 3)


def _make_tmp_dir():
    global tmp_dir
    global zip_dir

    if os.path.isdir(tmp_dir) == False:
        os.mkdir(tmp_dir)

    if os.path.isdir(zip_dir) == False:
        os.mkdir(zip_dir)


def make_dataset_dirs(dataset_name):
    global tmp_dir
    _make_tmp_dir()

    dir = tmp_dir + "/{}".format(dataset_name)

    if os.path.isdir(dir) == False:
        os.mkdir(dir)


def bar_progress(current, total, width=80):
    import sys

    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

dataset = GetData("")

dataset["train_data"] = dataset["train_data"].astype(np.float32)
dataset["test_data"] = dataset["test_data"].astype(np.float32)

# dataset["train_data"] / 255.0
# dataset["test_data"] / 255.0

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

input_shape=[32,32,3] # default size for cifar10
num_classes=10 # default class number for cifar10
num_filters = 16

inputs = Input(shape=(32, 32, 3))
x = Conv2D(num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4))(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# First stack

# Weight layers
y = Conv2D(num_filters,
              kernel_size=3,
              strides=1,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Conv2D(num_filters,
              kernel_size=3,
              strides=1,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(y)
y = BatchNormalization()(y)

# Overall residual, connect weight layer and identity paths
x = tf.keras.layers.add([x, y])
x = Activation('relu')(x)


# Second stack

# Weight layers
num_filters = 32 # Filters need to be double for each stack
y = Conv2D(num_filters,
              kernel_size=3,
              strides=2,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Conv2D(num_filters,
              kernel_size=3,
              strides=1,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(y)
y = BatchNormalization()(y)

# Adjust for change in dimension due to stride in identity
x = Conv2D(num_filters,
              kernel_size=1,
              strides=2,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(x)

# Overall residual, connect weight layer and identity paths
x = tf.keras.layers.add([x, y])
x = Activation('relu')(x)


# Third stack

# Weight layers
num_filters = 64
y = Conv2D(num_filters,
              kernel_size=3,
              strides=2,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Conv2D(num_filters,
              kernel_size=3,
              strides=1,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(y)
y = BatchNormalization()(y)

# Adjust for change in dimension due to stride in identity
x = Conv2D(num_filters,
              kernel_size=1,
              strides=2,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(x)

# Overall residual, connect weight layer and identity paths
x = tf.keras.layers.add([x, y])
x = Activation('relu')(x)

 # Final classification layer.
pool_size = int(np.amin(x.shape[1:3]))
x = AveragePooling2D(pool_size=pool_size)(x)
y = Flatten()(x)
outputs = Dense(num_classes,
                activation='softmax',
                kernel_initializer='he_normal')(y)

# Instantiate model.
model = Model(inputs=inputs, outputs=outputs)

EPOCHS = 5
BS = 512

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(dataset["train_data"], dataset["train_labels"], epochs=EPOCHS, batch_size=BS)
model.summary()
res = model.evaluate(dataset["test_data"], dataset["test_labels"])
print("Model1 has an accuracy of {0:.2f}%".format(res[1] * 100))

converter = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_dataset():
    for i in range(500):
        yield [np.array(dataset["train_data"][i:i+1])]
                # yield [np.array(dataset["train_data"][i:i+1])]


print(np.array(dataset["train_data"][0]).ndim)

converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_new_quantizer=True

tflite_quant_model = converter.convert()

with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)