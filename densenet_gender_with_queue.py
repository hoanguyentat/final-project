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
x, labels = read_and_decode_tfrecords(tfrecord_fn, total_epochs)
# train_x, test_x = color_preprocessing(train_x, test_x)
labels = tf.one_hot(labels, class_num_gender)

print("Modeling....")
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')


logits = DenseNet(x=x, nb_blocks=nb_blocks, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name="cost")

# l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
# l2 weight decay loss
costs = []
for var in tf.trainable_variables():
    costs.append(tf.nn.l2_loss(var))
    tf.summary.histogram(var.op.name, var)
l2_loss = tf.add_n(costs)

# regular_loss = cost + l2_loss * weight_decay

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True, name="optimizer")
train = optimizer.minimize(cost + l2_loss * weight_decay, name='train')


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
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
parameter_log = "growth_k = %d, init_learning_rate = %f, batch_size = %d, weight_decay = %f, number_of_block = %d, image_size = %d \n" % (growth_k, init_learning_rate, batch_size, weight_decay, nb_blocks, image_size)
with open('logs-gender.txt', 'a') as f:
    f.write(parameter_log)
print("Modeling done, starting training...")
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    ckpt = tf.train.get_checkpoint_state('./model-gender-new')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    #start input enqueue threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess= sess, coord=coord)
    epoch_learning_rate = init_learning_rate

    summary_writer = tf.summary.FileWriter('./logs-gender-new', sess.graph)
    
    try:
        step = 0
        while not coord.should_stop():
            start_time = time.time()
            if step == (total_epochs * 0.2) or step == (total_epochs * 0.4) or step == (total_epochs * 0.6) or step == (total_epochs * 0.8):
                epoch_learning_rate = epoch_learning_rate / 10
            train_feed_dict = {
                learning_rate: epoch_learning_rate,
                training_flag: True
            }
            _, train_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
            dur_time = time.time() - start_time

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
            tf.add_to_collection('train_summary', train_summary)
            # test_acc, test_loss, test_summary = Evaluate(sess, test_x, test_y)

           
            summary_writer.add_summary(summary=train_summary, global_step=step)
            # summary_writer.add_summary(summary=test_summary, global_step=step)
            # summary_writer.flush()
            step += 1
            log_line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, time: %ss \n" % (
                step, total_epochs, train_loss, train_acc, str(dur_time))
            print(log_line)
            with open('logs-gender.txt', 'a') as f:
                f.write(log_line)
            saver.save(sess=sess, save_path='./model-gender-new/dense.ckpt')
    except tf.errors.OutOfRangeError as e:
        print("Done training for %d epochs, %d steps." % (total_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
