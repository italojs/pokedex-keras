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
	def __init__(self, path):
		self.path_train = f"{path}/train"
		self.path_test = f"{path}/test"

		self.len_train = len(list(paths.list_images(self.path_train)))
		self.len_test = len(list(paths.list_images(self.path_test)))

		self.len_classes = len(os.listdir(self.path_train))
	
	def generator(self, batch_size):
		train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			data_format="channels_last")

		return train_datagen.flow_from_directory('./output/train/',target_size=(224,224), class_mode='categorical', batch_size=batch_size)

	def save_labels(self, le_path):
		f = open(le_path, "wb")
		f.write(pickle.dumps(self.labelEncoder))
		f.close()