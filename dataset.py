from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

class Dataset():

	def __init__(self, trainX=[], testX=[], trainY=[], testY=[], le=None, shape=(0,0,0)):
		self.trainX = trainX	
		self.testX = testX
		self.trainY = trainY
		self.testY = testY
		self.labelEncoder = le
		self.shape = shape
	
	def generator(self, batch_size):
		# TODO: set it optional		
		aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
			width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
			horizontal_flip=True, fill_mode="nearest")
		return aug.flow(self.trainX, self.trainY, batch_size=batch_size)
	
	def save_labels(self, le_path):
		f = open(le_path, "wb")
		f.write(pickle.dumps(self.labelEncoder))
		f.close()

def split(X, y):
	stratSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
	for train_index, test_index in stratSplit.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
	
	return X_train, X_test, y_train, y_test

def dataset_factory(dataset_path, input_shape, test_size=0.25, random_state=42):
	imagePaths = list(paths.list_images(dataset_path))
	data = []
	labels = []

	for imagePath in imagePaths:
		label = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (input_shape[0], input_shape[1]))

		data.append(image)
		labels.append(label)

	data = np.array(data, dtype="float") / 255.0
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	labels = np_utils.to_categorical(labels)

	(trainX, testX, trainY, testY) = split(data, labels)

	return Dataset(trainX, testX, trainY, testY, le, input_shape)

