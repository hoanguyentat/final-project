import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import os
from scipy import ndimage
import cv2
from utils import load_data
from utils import color_preprocessing
from utils import data_augmentation
import numpy as np

growth_k = 24
data_path = 'data/wiki_db_32.mat'
# Number of (dense block + Transition Layer)
nb_block = 2
init_learning_rate = 1e-4
# AdamOptimizer epsilon
epsilon = 1e-4

dropout_rate = 0.2
class_num = 2
image_size = 32
img_channels = 3

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 100

test_iteration = 60
test_batch_size = 100
total_epochs = 50


# def reformat(dataset, labels):
#     dataset = dataset.reshape(-1, image_size * image_size * img_channels).astype(np.float32)
#     labels = (np.arange(class_num) == labels[:, None]).astype(np.float32)
#     return dataset, labels

def reformat(labels):
    labels = (np.arange(class_num) == labels[:, None]).astype(np.float32)
    # labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    return labels


def randomize(dataset, gender, age):
    permutation = np.random.permutation(age.shape[0])
    shuffle_dataset = dataset[permutation, :, :, :]
    shuffle_labels_gender = gender[permutation]
    shuffle_labels_age = age[permutation]
    return shuffle_dataset, shuffle_labels_gender, shuffle_labels_age


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        return network


def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

    # The stride value does not matter    It is global average pooling without tflearn
    # return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x):
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)


        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        # x = tf.reshape(x, [-1, 10])
        return x


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
    label = tf.placeholder(tf.float32, shape=[None, class_num])

    training_flag = tf.placeholder(tf.bool)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(tf.global_variables())

def load_images_from_folder(fName):
    images = []
    for image in os.listdir(fName):
        img = cv2.imread(os.path.join(fName, image))
        if img is not None:
            img = cv2.resize(img, (32, 32))
            images.append(img)
    return images


def predict(folder, file):
    if not os.path.exists(folder):
        print("File is not exist...")
    images = load_images_from_folder(folder)
    feed_dict = {x: images, training_flag: True}
    with tf.Session(graph=graph) as sess:
        ckpt = tf.train.get_checkpoint_state(file)
        print(ckpt)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint found...")
        predict_label = tf.arg_max(logits, 1)
        print(predict_label.eval(feed_dict=feed_dict, session=sess))


if __name__ == '__main__':
    # restore_graph("./model-gender")
    predict('data/wiki_crop/00', './model-gender')
