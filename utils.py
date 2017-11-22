from scipy.io import loadmat
from datetime import datetime
import os
import numpy as np
import random
from parameters import *


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]


def mk_dir(dir):
    try:
        os.mkdir(dir)
    except OSError:
        pass


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [image_size, image_size], 4)
    return batch


def reformat(labels, class_num):
    labels = (np.arange(class_num) == labels[:, None]).astype(np.float32)
    # labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    return labels


# def reformat(dataset, labels):
#     dataset = dataset.reshape(-1, image_size * image_size * img_channels).astype(np.float32)
#     labels = (np.arange(class_num) == labels[:, None]).astype(np.float32)
#     return dataset, labels


def randomize(dataset, gender, age):
    permutation = np.random.permutation(age.shape[0])
    shuffle_dataset = dataset[permutation, :, :, :]
    shuffle_labels_gender = gender[permutation]
    shuffle_labels_age = age[permutation]
    return shuffle_dataset, shuffle_labels_gender, shuffle_labels_age


def main():
    # full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta("data/wiki_crop/wiki.mat", "wiki")
    print("----------------Load data----------------")
    image_dataset, labels_gender, labels_age, _, img_size, _ = load_data("data/wiki_db_32.mat")
    # print(max(label_age), min(label_age), Counter(label_age))
    print(img_size)
    print(image_dataset.shape)
    print(labels_gender.shape)
    print("Max age: ", max(labels_age))
    unique, counts = np.unique(labels_age, return_counts=True)
    print(unique)
    labels = (np.arange(101) == labels_age[:, None]).astype(np.float32)
    # print(labels)
    print(labels.shape)
    print("----------------get Meta------------------")
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta("data/wiki_crop/wiki.mat",
                                                                                       "wiki")
    print(full_path[0])
    print(gender[0], age[0])
    print(photo_taken[0])


if __name__ == '__main__':
    main()
