# [truong] define input for estimator and serving
from word_generator import generate_data, get_augmenter
import tensorflow as tf
import functools

def train_input_fn(params, coco_imgs):
    batch_size = params.batch_size
    epochs = params.num_epochs

    # Creat char map
    shapes = (([None, params.img_h, params.img_w, params.img_dim], [None, None], [None]), [None, None])
    types = ((tf.float32, tf.int32, tf.int32), tf.int32)
    # defaults = ((0.0, 0, 'PAD', 0), 'O')

    aug = get_augmenter()

    print('COCO lens = ', len(coco_imgs))

    dataset = tf.data.Dataset.from_generator(functools.partial(generate_data, batch_size, epochs, aug, coco_imgs, (params.img_w, params.img_h, params.img_dim)),
                             output_types=types, output_shapes=shapes)
    # dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(64)

    return dataset


def val_input_fn(params):
    batch_size = params.batch_size
    epochs = params.num_epochs

    # Creat char map
    shapes = (([None, params.img_h, params.img_w, params.img_dim], [None, None], [None]), [None, None])
    types = ((tf.float32, tf.int32, tf.int32), tf.int32)

    dataset = tf.data.Dataset.from_generator(functools.partial(generate_data, batch_size, 5, None, None, (params.img_w, params.img_h, params.img_dim)),
                             output_types=types, output_shapes=shapes)
    # dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(64)

    return dataset


def serving_input_fn():
    image = tf.placeholder(dtype=tf.float32, shape=[None, 80, 100, 1], name='image')
    receiver_tensors = {'image':image}
    return tf.estimator.export.ServingInputReceiver(image, receiver_tensors)