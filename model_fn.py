"""Define the model."""
import os
import tensorflow as tf
# from tf_metrics import precision, recall, f1
from pathlib import Path

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib import layers

# Upgrade to 2.0
from tensorflow.python.ops import lookup_ops, array_ops

from tensorflow.keras.layers import LSTM, Bidirectional


START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

def emb_initialize():
    def intialize(shape, dtype, partition_info=None):
        embedding = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        embedding = tf.nn.l2_normalize(embedding, -1)
        return embedding
    return intialize


from tensorflow.contrib import slim


def conv2d(layer, name, n_filters, trainable, k_size=3):
    return tf.layers.conv2d(layer, n_filters, kernel_size=(k_size, k_size),
                            activation=tf.nn.relu, padding=self.conv_padding, name=name, trainable=trainable,
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            use_bias=True)
def vgg_16(inputs,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    end_points = {}
    with tf.variable_scope(scope):
        # Conv1
        with tf.variable_scope('conv1'):
            net = tf.layers.conv2d(inputs, 64, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv1_1')
            net = tf.layers.conv2d(net, 64, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv1_2')

        # Pool 1
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='SAME')

        # Conv2
        with tf.variable_scope('conv2'):
            net = tf.layers.conv2d(net, 128, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv2_1')
            net = tf.layers.conv2d(net, 128, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv2_2')

        end_points['conv2_2'] = net

        # Pool 2
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='SAME')

        # Conv3
        with tf.variable_scope('conv3'):
            net = tf.layers.conv2d(net, 256, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv3_1')
            net = tf.layers.conv2d(net, 256, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv3_2')
            net = tf.layers.conv2d(net, 256, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv3_3')

        end_points['conv3_3'] = net

        # Pool 3
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='SAME')

        # Conv 4
        with tf.variable_scope('conv4'):

            net = tf.layers.conv2d(net, 512, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv4_1')
            net = tf.layers.conv2d(net, 512, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv4_2')
            net = tf.layers.conv2d(net, 512, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv4_3')

        end_points['conv4_3'] = net

        # Pool 4
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='SAME')

        # Conv 5
        with tf.variable_scope('conv5'):

            net = tf.layers.conv2d(net, 512, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv5_1')
            net = tf.layers.conv2d(net, 512, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv5_2')
            net = tf.layers.conv2d(net, 512, [3,3], activation=tf.nn.relu, padding='SAME', kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), use_bias=True, name='conv5_3')

        end_points['conv5_3'] = net

        # Pool 5
        net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='SAME')
        return net, end_points



def build_encoder(images, is_training, params):
    with tf.variable_scope('encoder'):
        # Conv1
        W1 = tf.get_variable(dtype=tf.float32, shape=[3,3,1,64], name='weights_conv1', initializer=tf.glorot_uniform_initializer())
        b1 = tf.get_variable(dtype=tf.float32, shape=[64], name='biases_conv1', initializer=tf.zeros_initializer())

        conv1 = tf.nn.bias_add(tf.nn.conv2d(images, W1, strides=[1,1,1,1], padding='SAME'), b1)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        # Conv2
        W2 = tf.get_variable(dtype=tf.float32, shape=[3,3,64,128], name='weights_conv2', initializer=tf.glorot_uniform_initializer())
        b2 = tf.get_variable(dtype=tf.float32, shape=[128], name='biases_conv2', initializer=tf.zeros_initializer())

        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, W2, strides=[1,1,1,1], padding='SAME'), b2)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        # Conv3
        W3 = tf.get_variable(dtype=tf.float32, shape=[3,3,128,256], name='weights_conv3', initializer=tf.glorot_uniform_initializer())
        b3 = tf.get_variable(dtype=tf.float32, shape=[256], name='biases_conv3', initializer=tf.zeros_initializer())

        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, W3, strides=[1,1,1,1], padding='SAME'), b3)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)

        # Conv4
        W4 = tf.get_variable(dtype=tf.float32, shape=[3,3,256,256], name='weights_conv4', initializer=tf.glorot_uniform_initializer())
        b4 = tf.get_variable(dtype=tf.float32, shape=[256], name='biases_conv4', initializer=tf.zeros_initializer())

        conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME'), b4)
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)

        pool3 = tf.nn.max_pool(conv4, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

        # Conv5
        W5 = tf.get_variable(dtype=tf.float32, shape=[3,3,256,512], name='weights_conv5', initializer=tf.glorot_uniform_initializer())
        b5 = tf.get_variable(dtype=tf.float32, shape=[512], name='biases_conv5', initializer=tf.zeros_initializer())

        conv5 = tf.nn.bias_add(tf.nn.conv2d(pool3, W5, strides=[1,1,1,1], padding='SAME'), b5)
        conv5 = tf.nn.relu(conv5)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)

        # Conv6
        W6 = tf.get_variable(dtype=tf.float32, shape=[3,3,512,512], name='weights_conv6', initializer=tf.glorot_uniform_initializer())
        b6 = tf.get_variable(dtype=tf.float32, shape=[512], name='biases_conv6', initializer=tf.zeros_initializer())

        conv6 = tf.nn.bias_add(tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='SAME'), b6)
        conv6 = tf.nn.relu(conv6)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training)
        pool5 = tf.nn.max_pool(conv6, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

        # Conv7
        W7 = tf.get_variable(dtype=tf.float32, shape=[2,2,512,512], name='weights_conv7', initializer=tf.glorot_uniform_initializer())
        b7 = tf.get_variable(dtype=tf.float32, shape=[512], name='biases_conv7', initializer=tf.zeros_initializer())

        conv7 = tf.nn.bias_add(tf.nn.conv2d(pool5, W7, strides=[1,1,1,1], padding='VALID'), b7)
        conv7 = tf.nn.relu(conv7)

        out_conv = tf.squeeze(conv7, axis=1)

        cell_fw = tf.nn.rnn_cell.LSTMCell(params.encoder_lstm_hidden)
        cell_bw = tf.nn.rnn_cell.LSTMCell(params.encoder_lstm_hidden)

        out, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, out_conv, dtype=tf.float32)
        out = tf.concat(out, axis=-1)

        final_state_c = tf.concat((state_fw.c, state_bw.c), 1)
        final_state_h = tf.concat((state_fw.h, state_bw.h), 1)
        state = LSTMStateTuple(c=final_state_c, h=final_state_h)

        return out, state


def build_encoder_v1(images, is_training, params):
    # Experimental is w x h = 150 x 120
    vgg_res, end_points = vgg_16(images)

    if is_training == True:
        batch_size = params.batch_size
    else:
        batch_size = 1

    # Add positional embedding
    _, h, w, _ = vgg_res.shape.as_list()
    x, y = tf.meshgrid(tf.range(w), tf.range(h))
    w_loc = tf.one_hot(x, depth=w)
    h_loc = tf.one_hot(y, depth=h)
    loc = tf.concat([h_loc, w_loc], 2)

    # Batch size cheat
    loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
    out = tf.concat([vgg_res, loc], 3)

    # Flatten positional
    features_h = array_ops.shape(out)[1]
    feature_size = out.get_shape().dims[3].value
    out = tf.reshape(out, [batch_size, -1, feature_size])

    return out, None


def build_train_decoder(decoder_in, encoder_out, state, seq_length, params):
    # Declare embedding:
    # with tf.device('/cpu:0'):
    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable('embedding', initializer=emb_initialize(), shape=[params.max_char_vocab, params.char_embedding_dim], dtype=tf.float32)

    batch_size = tf.shape(encoder_out)[0]
    start_token = tf.ones([batch_size], dtype=tf.int32) * START_TOKEN
    # Create training params
    # Add start token and update seq length
    train_inp = tf.concat([tf.expand_dims(start_token, 1), decoder_in], 1)
    max_length = tf.reduce_max((seq_length+1))

    # Create embedding lookup for training
    train_embedding = tf.nn.embedding_lookup(embedding, train_inp)
    train_helper = tf.contrib.seq2seq.TrainingHelper(train_embedding, seq_length+1)


    with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
        # Build attention
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=params.attention_units, memory=encoder_out)
        decoder_cell = tf.nn.rnn_cell.LSTMCell(params.encoder_lstm_hidden*2)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                attention_layer_size=params.attention_layer_size)
        project_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, params.max_char_vocab)
        decoder_initial_state = attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

        if state is not None:
            decoder_initial_state = decoder_initial_state.clone(cell_state=state)

        decoder = tf.contrib.seq2seq.BasicDecoder(project_cell, train_helper, decoder_initial_state)
        train_out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True,
                                                            maximum_iterations=max_length)

    return train_out


def build_infer_decoder(encoder_out, state, params):
    # Declare embedding:
    with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable('embedding', initializer=emb_initialize(), shape=[params.max_char_vocab, params.char_embedding_dim], dtype=tf.float32)

    batch_size = tf.shape(encoder_out)[0]
    start_token = tf.ones([batch_size], dtype=tf.int32) * START_TOKEN
    # Create training params
    # Add start token and update seq length
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.to_int32(start_token), end_token=1)

    with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=params.attention_units, memory=encoder_out)
        decoder_cell = tf.nn.rnn_cell.LSTMCell(params.encoder_lstm_hidden*2)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                attention_layer_size=params.attention_layer_size)
        project_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, params.max_char_vocab)
        decoder_initial_state = attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        if state is not None:
            decoder_initial_state = decoder_initial_state.clone(cell_state=state)

        decoder = tf.contrib.seq2seq.BasicDecoder(project_cell, pred_helper, decoder_initial_state)
        pred_out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True,
                                                            maximum_iterations=32)

    return pred_out


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if isinstance(features, dict):  # For serving
        print(features)
        images = features['feature']
    else:
        (images, decoder_in, seq_length) = features

    if params.use_vgg == 0:
        encoder_out, state = build_encoder(images, is_training, params)
    else:
        print("Use vgg to build encoder")
        encoder_out, state = build_encoder_v1(images, is_training, params)


    # Perform main business
    if mode == tf.estimator.ModeKeys.PREDICT:
        pred_out = build_infer_decoder(encoder_out, state, params)

        export_outputs = {'sample_id': tf.estimator.export.PredictOutput(pred_out.sample_id)}
        predictions = {'pred_out': pred_out.sample_id}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
    else:
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_out = build_train_decoder(decoder_in, encoder_out, state, seq_length, params)
            # Train
            global_step = tf.train.get_global_step()
            mask = tf.cast(tf.sequence_mask(seq_length+1), tf.float32)
            loss = tf.contrib.seq2seq.sequence_loss(train_out.rnn_output, labels, weights=mask,
                                                    average_across_timesteps=True, average_across_batch=True)

            learning_rate = tf.train.exponential_decay(params.learning_rate, global_step, params.learning_rate_step,
                                                    params.learning_rate_decay, staircase=True, name='exponential_learning_rate')

            update_global_step = tf.assign(global_step, global_step + 1, name = 'update_global_step')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group(train_op, update_global_step))
        else:
            pred_out = build_infer_decoder(encoder_out, state, params)

            mask = tf.cast(tf.sequence_mask(seq_length+1), tf.float32)
            loss = tf.contrib.seq2seq.sequence_loss(pred_out.rnn_output, labels, weights=mask,
                                                    average_across_timesteps=True, average_across_batch=True)

            return tf.estimator.EstimatorSpec(mode, loss=loss)
