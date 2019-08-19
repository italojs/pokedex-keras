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
	def __init__(self, path, target_size=(224,224)):
		self.__path_train = f"{path}/train"
		self.__path_test = f"{path}/test"

		self.__paths_train = list(paths.list_images(self.__path_train))
		self.__paths_test = list(paths.list_images(self.__path_test))
		
		self.__target_size = target_size
		
		self.len_train = len(self.__paths_train)
       
		self.len_classes = len(os.listdir(self.__path_train))
		
		labels = [path.split(os.path.sep)[-2] for path in self.__paths_test]
		
		self.le = LabelEncoder()
		self.labels = self.le.fit_transform(labels)

	
	def generator(self, batch_size):
		train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			data_format="channels_last")

		return train_datagen.flow_from_directory('./output/train/',target_size=self.__target_size, class_mode='categorical', batch_size=batch_size)

	def save_labels(self, le_path):
		f = open(le_path, "wb")
		f.write(pickle.dumps(self.le))
		f.close()

	def load_test_data(self):
		data = []

		for imagePath in self.__paths_test:
			label = imagePath.split(os.path.sep)[-2]
			image = cv2.imread(imagePath)
			image = cv2.resize(image, (self.__target_size[0], self.__target_size[1]))

			data.append(image)

		return np.array(data, dtype="float") / 255.0