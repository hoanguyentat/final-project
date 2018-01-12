import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta
from random import shuffle
import glob
import tensorflow as tf
import sys


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
												 "and creates database for training.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--output", "-o", type=str, help="path to output database mat file")
	parser.add_argument("--db", type=str, default="wiki",
						help="dataset; wiki or imdb")
	parser.add_argument("--img_size", type=int, default=32,
						help="output image size")
	parser.add_argument("--min_score", type=float, default=1.0,
						help="minimum face_score")
	parser.add_argument("--train_fraction", type=float, default=0.8,
						help="fraction of train dataset")
	args = parser.parse_args()
	return args


def main():
	args = get_args()
	output_path = args.output
	db = args.db
	img_size = args.img_size
	min_score = args.min_score
	fr = args.train_fraction

	root_path = "data/{}_crop/".format(db)
	mat_path = root_path + "{}.mat".format(db)
	full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

	out_genders = []
	out_ages = []
	out_imgs = []

	for i in tqdm(range(len(face_score))):
		if face_score[i] < min_score:
			continue

		if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
			continue

		if ~(0 <= age[i] <= 100):
			continue

		if np.isnan(gender[i]):
			continue

		out_genders.append(int(gender[i]))
		out_ages.append(age[i])
		img = cv2.imread(root_path + str(full_path[i][0]))
		out_imgs.append(cv2.resize(img, (img_size, img_size)))

	number_of_train = int(fr * len(out_imgs))
	train_images = out_imgs[0:number_of_train]
	labels_train_gender = out_genders[0: number_of_train]
	labels_train_age = out_ages[0:number_of_train]

	valid_images = out_imgs[number_of_train:]
	labels_valid_gender = out_genders[number_of_train:]
	labels_valid_age = out_ages[number_of_train:]

	tfrecord_train = 'train_' + str(img_size) + '.tfrecords'
	tfrecord_valid = 'valid_' + str(img_size) + '.tfrecords'
	create_tfrecord(tfrecord_train, train_images,labels_train_age, labels_train_gender)
	create_tfrecord(tfrecord_valid, valid_images,labels_valid_age, labels_valid_gender)

def create_tfrecord(fn, data, labels_age, labels_gender):
	writer = tf.python_io.TFRecordWriter(fn)
	for i in range(len(data)):
		if not i % 1000:
			print("Converting {}/{}".format(i, len(data)))
			sys.stdout.flush()
		feature = {
			'label_age': _int64_feature(labels_age[i]),
			'label_gender': _int64_feature(labels_gender[i]),
			'image': _bytes_feature(data[i].tostring())
		}
		record = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(record.SerializeToString())
	writer.close()
	sys.stdout.flush()

if __name__ == '__main__':
	main()
