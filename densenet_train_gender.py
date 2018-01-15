import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
# from utils import color_preprocessing
# from utils import data_augmentation
from utils import *
from parameters import *
from densenet import *
import time


def Evaluate(sess, test_x, test_y):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    test_iteration = int(test_y.shape[0] / test_batch_size)
    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + test_batch_size]
        test_batch_y = test_y[test_pre_index: test_pre_index + test_batch_size]
        test_pre_index = test_pre_index + test_batch_size

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / test_iteration
        test_acc += acc_ / test_iteration

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


# get data from file
train_image, labels_gender, labels_age, _, _, _ = load_data(data_path)
labels_gender = reformat(labels_gender, class_num_gender)
labels_age = reformat(labels_age, class_num_age)

# Divide data for train dataset and test dataset
nbtrain = int(train_image.shape[0] * train_fraction)
train_x = train_image[0:nbtrain, :, :, :]
train_y = labels_gender[0:nbtrain]
test_x = train_image[nbtrain:, :, :, :]
test_y = labels_gender[nbtrain:]
print(train_x.shape, train_y.shape)

train_x, test_x = color_preprocessing(train_x, test_x)

print("Modeling....")
# Variables
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
print("x: ", x.shape)
label = tf.placeholder(tf.float32, shape=[None, class_num_gender])
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=x, nb_blocks=nb_blocks, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits), name="cost")

# l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
# l2 weight decay loss
costs = []
for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
    tf.summary.histogram(var.op.name, var)
l2_loss = tf.add_n(costs)

# regular_loss = cost + l2_loss * weight_decay

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True,
                                       name="optimizer")
train = optimizer.minimize(cost + l2_loss * weight_decay, name='train')

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.add_to_collection('training_flag', training_flag)
tf.add_to_collection('x', x)
tf.add_to_collection('image_size', image_size)
tf.add_to_collection('img_channels', img_channels)
tf.add_to_collection('logits', logits)
tf.add_to_collection('learning_rate', learning_rate)
tf.add_to_collection('cost', cost)
tf.add_to_collection('optimizer', optimizer)
tf.add_to_collection('train', train)
tf.add_to_collection('l2_loss', l2_loss)
tf.add_to_collection('correct_prediction', correct_prediction)
tf.add_to_collection('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

# Start train
parameter_log = "growth_k = %d, init_learning_rate = %f, batch_size = %d, weight_decay = %f, number_of_block = %d, image_size = %d \n" % (
growth_k, init_learning_rate, batch_size, weight_decay, nb_blocks, image_size)
with open('logs-gender.txt', 'a') as f:
    f.write(parameter_log)
print("Modeling done, starting training...")
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    ckpt = tf.train.get_checkpoint_state('./model-gender-new')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    epoch_learning_rate = init_learning_rate

    summary_writer = tf.summary.FileWriter('./logs-gender-new', sess.graph)

    for epoch in range(1, total_epochs + 1):
        if epoch == (total_epochs * 0.2) or epoch == (total_epochs * 0.4) or epoch == (total_epochs * 0.6) or epoch == (
            total_epochs * 0.8):
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0
        iteration = int(train_y.shape[0] / batch_size)

        # start time of epoch
        start_time = time.time()
        for step in range(iteration):
            if pre_index + batch_size < train_y.shape[0]:
                batch_x = train_x[pre_index: pre_index + batch_size]
                batch_y = train_y[pre_index: pre_index + batch_size]
            else:
                batch_x = train_x[pre_index:]
                batch_y = train_y[pre_index:]

            # randomize data
            batch_x = data_augmentation(batch_x)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }
            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += batch_size

            # Buoc step phai so sanh voi iteration - 1
            if step == iteration - 1:
                train_loss /= iteration  # average loss
                train_acc /= iteration  # average accuracy

                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                tf.add_to_collection('train_summary', train_summary)
                test_acc, test_loss, test_summary = Evaluate(sess, test_x, test_y)

                dur_time = time.time() - start_time
                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                summary_writer.add_summary(summary=test_summary, global_step=epoch)
                # summary_writer.flush()

                log_line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f, time: %ss \n" % (
                    epoch, total_epochs, train_loss, train_acc, test_loss, test_acc, str(dur_time))
                print(log_line)
                with open('logs-gender.txt', 'a') as f:
                    f.write(log_line)
        saver.save(sess=sess, save_path='./model-gender-new/dense.ckpt')
