import tensorflow as tf
import os
import cv2
import json

image_size = 32
img_channels = 3

path_full = []
def load_images_from_folder(fName):
    images = []
    for image in os.listdir(fName):
        img = cv2.imread(os.path.join(fName, image))
        if img is not None:
            path_full.append(img)
            img = cv2.resize(img, (32, 32))
            images.append(img)
    return images


def predict(folder, file):
    if not os.path.exists(folder):
        print("File is not exist...")
    images = load_images_from_folder(folder)
    dic = {}
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(file)
        print(ckpt)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            saver.restore(sess, ckpt.model_checkpoint_path)
            x = tf.get_collection('x')[0]
            training_flag = tf.get_collection('training_flag')[0]
            logits = tf.get_collection('logits')[0]
            print(type(x))
            feed_dict = {x: images, training_flag: True}
        else:
            print("No checkpoint found...")
        predict_label = tf.arg_max(logits, 1)
        result = predict_label.eval(feed_dict=feed_dict, session=sess)
        for i in len(path_full):
            dic[path_full[i]] = "Male" if result[i] else "Female"
        print(dic)
    

def evulate():
    valid = []
    with open('data/test/img20131.json') as data_file:
        imgs = data_file.read()
        print(type(imgs))
        dic = json.loads(imgs)
        print(type(dic))
        for (key, val) in enumerate(dic):
            if dic[val] == "male":
                valid.append(1)
            else:
                valid.append(0)
    print(valid)
if __name__ == '__main__':
    # restore_graph("./model-gender")
    # predict('data/test/img20131/', './model-gender')
    evulate()