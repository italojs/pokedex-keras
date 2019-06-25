from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from imutils import paths
import numpy as np
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
		print("[INFO] Image data generator...")		
		aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
			width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
			horizontal_flip=True, fill_mode="nearest")
		return aug.flow(self.trainX, self.trainY, batch_size=batch_size)
	
	def save_labels(self, le_path):
		print("[INFO] saving labels...")
		f = open(le_path, "wb")
		f.write(pickle.dumps(self.labelEncoder))
		f.close()

def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion) #should be int
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

def dataset_factory(dataset_path, input_shape, test_size=0.25, random_state=42):
	print("[INFO] loading images...")
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

	(trainX, testX, trainY, testY) = shuffle(data, labels, 3)

	return Dataset(trainX, testX, trainY, testY, le, input_shape)

