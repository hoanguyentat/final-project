import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import os
from scipy import ndimage
import cv2
import numpy as np

image_size = 32
img_channels = 3

training_flag = tf.placeholder(tf.bool)
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

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(file)
        print(ckpt)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            saver.restore(sess, ckpt.model_checkpoint_path)
            x = tf.get_collection('x')
            # training_flag = tf.get_collection('training_flag')[0]
            print(type(x))
            feed_dict = {x: images, training_flag: True}
        else:
            print("No checkpoint found...")
        # predict_label = tf.arg_max(logits, 1)
        # print(predict_label.eval(feed_dict=feed_dict, session=sess))


if __name__ == '__main__':
    # restore_graph("./model-gender")
    predict('data/wiki_crop/00', './model-gender')
