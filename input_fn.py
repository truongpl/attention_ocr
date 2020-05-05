# [truong] define input for estimator and serving
from word_generator import generate_data
import tensorflow as tf
import functools

def train_input_fn(params):
    batch_size = params.batch_size
    epochs = params.num_epochs

    # Creat char map
    shapes = (([None, 32, 128, 1], [None, None], [None]), [None, None])
    types = ((tf.float32, tf.int32, tf.int32), tf.int32)
    # defaults = ((0.0, 0, 'PAD', 0), 'O')

    dataset = tf.data.Dataset.from_generator(functools.partial(generate_data, batch_size, epochs),
                             output_types=types, output_shapes=shapes)
    # dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(64)

    return dataset


def val_input_fn(params):
    batch_size = params.batch_size
    epochs = params.num_epochs

    # Creat char map
    shapes = (([None, 32, 128, 1], [None, None], [None]), [None, None])
    types = ((tf.float32, tf.int32, tf.int32), tf.int32)

    dataset = tf.data.Dataset.from_generator(functools.partial(generate_data, batch_size, 5),
                             output_types=types, output_shapes=shapes)
    # dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(64)

    return dataset