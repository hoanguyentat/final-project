import tensorflow as tf
import os
import cv2
import json

image_size = 96
img_channels = 3

path_full = []
def load_images_from_folder(fName):
    images = []
    for image in os.listdir(fName):
        img = cv2.imread(os.path.join(fName, image))
        if img is not None:
            path_full.append(image)
            img = cv2.resize(img, (image_size, image_size))
            images.append(img)
    # print(path_full)
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
            # print(type(x))
            feed_dict = {x: images, training_flag: True}
        else:
            print("No checkpoint found...")
        predict_label = tf.arg_max(logits, 1)
        result = predict_label.eval(feed_dict=feed_dict, session=sess)
        print(result)
        for i in range(len(path_full)):
            dic[path_full[i]] = "male" if result[i] else "female"
        print(dic)
        dic = json.dumps(dic, sort_keys=True)
    with open("result.json", "w") as file:
        json.dump(dic, file)
    
if __name__ == '__main__':
    # restore_graph("./model-gender")
    predict('data/test/', './model-gender-new')