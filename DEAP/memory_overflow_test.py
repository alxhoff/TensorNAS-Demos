import gc
import tracemalloc, sys, linecache, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading

EPOCHS = 5
BS = 512
TEST_LOOPS = 10000


def start_tracemalloc():
    tracemalloc.start()


def display_top(snapshot, key_type="lineno", limit=5):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def display_snapshot():
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)


def create_model() -> keras.models.Model:
    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), padding="valid")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation=tf.nn.relu)(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(10, activation=tf.nn.softmax)(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BS)


def do_model_in_thread(train_images, train_labels, model_list):
    model = create_model()
    train_model(model=model, train_images=train_images, train_labels=train_labels)
    model_list[0] = model


def do_model(train_images, train_labels):
    model = create_model()
    train_model(model=model, train_images=train_images, train_labels=train_labels)

    return model


def dump_garbage():
    print("\nGARBAGE OBJECTS:")
    sym_tens = [x for x in gc.garbage if x.__class__.__name__ is 'SymbolicTensor']
    print(len(sym_tens))
    for x in sym_tens:
        s = str(x)
        if len(s) > 80: s = s[:77] + '...'
        print(type(x), "\n  ", s)


def main() -> int:
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape(
        train_images.shape[0], train_images.shape[1], train_images.shape[2], 1
    )
    train_images = train_images.astype(np.float32) / 255.0

    start_tracemalloc()
    gc.enable()
    gc.set_debug(gc.DEBUG_LEAK)

    for i in range(TEST_LOOPS):
        # model_list = [None] * 1
        # thread = threading.Thread(target=do_model, args=(train_images, train_labels, model_list))
        # thread.start()
        # thread.join()
        model = do_model(train_images, train_labels)
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        gc.collect()
        dump_garbage()
        display_snapshot()

        # objects = [x for x in gc.get_objects() if x.__class__.__name__ is 'SymbolicTensor']
        #
        # print("wait here")
        #
        # for x in objects:
        #     del x
        #
        # objects = [x for x in gc.get_objects() if x.__class__.__name__ is 'SymbolicTensor']

        print("wait here")


if __name__ == "__main__":
    sys.exit(main())
