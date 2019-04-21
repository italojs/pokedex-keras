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
		aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
			width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
			horizontal_flip=True, fill_mode="nearest")
			
		return aug.flow(self.trainX, self.trainY, batch_size=batch_size)
	
	def save_labels(self, le_path):
		# save the label encoder to disk
		f = open(le_path, "wb")
		f.write(pickle.dumps(self.labelEncoder))
		f.close()
		
def dataset_factory(dataset_path, input_shape, test_size=0.25, random_state=42):
	# grab the list of images in our dataset directory, then initialize
	# the list of data (i.e., images) and class images
	print("[INFO] loading images...")
	imagePaths = list(paths.list_images(dataset_path))
	data = []
	labels = []

	for imagePath in imagePaths:
		# extract the class label from the filename, load the image and
		# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
		label = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (input_shape[0], input_shape[1]))

		# update the data and labels lists, respectively
		data.append(image)
		labels.append(label)

	# convert the data into a NumPy array, then preprocess it by scaling
	# all pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0

	# encode the labels (which are currently strings) as integers and then
	# one-hot encode them
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	labels = np_utils.to_categorical(labels)

	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(
		data, labels, test_size=test_size, random_state=random_state)
	return Dataset(trainX, testX, trainY, testY, le, input_shape)

