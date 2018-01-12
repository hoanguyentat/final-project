import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from utils import *
from parameters import *

depth = 40


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        return network


def global_average_pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)


def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   # model experiences reasonably good training performance but poor validation and/or test performance
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


# def batch_normalization(x, training, scope):
#     output = batch_norm(x, scale=True, is_training=training, scope=scope, updates_collections=None)
#     return output

def tf_dropout(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def tf_relu(x):
    return tf.nn.relu(x)


def tf_average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def tf_max_pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x):
    return tf.layers.dense(inputs=x, units=class_num_gender, name='linear')


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope + '_batch1')
            x = tf_relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = tf_dropout(x, rate=dropout_rate, training=self.training)

            x = batch_normalization(x, training=self.training, scope=scope + '_batch2')
            x = tf_relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = tf_dropout(x, rate=dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope + '_batch1')
            x = tf_relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = tf_dropout(x, rate=dropout_rate, training=self.training)
            x = tf_average_pooling(x, pool_size=[2, 2], stride=2)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            # Run for bottle 0
            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        x = tf_max_pooling(input_x, pool_size=[3, 3], stride=2)

        layers_per_block = (depth - (self.nb_blocks + 1)) // self.nb_blocks

        for i in range(self.nb_blocks):
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=layers_per_block, layer_name='dense_' + str(i))
            if i != self.nb_blocks - 1:
                x = self.transition_layer(x, scope='trans_' + str(i))

        # x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        # x = self.transition_layer(x, scope='trans_1')

        # x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        # x = self.transition_layer(x, scope='trans_2')

        # x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        # x = self.transition_layer(x, scope='trans_3')

        # x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        x = batch_normalization(x, training=self.training, scope='linear_batch')
        x = tf_relu(x)
        x = global_average_pooling(x)
        x = flatten(x)
        x = Linear(x)

        return x
