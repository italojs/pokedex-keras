# import the necessary packages
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from model import Resnet50
from time import time
import numpy as np
import matplotlib

# matplotlib.use("Agg")

class NeuralNetwork:

	def __init__(self, epochs, batch_size,learn_rate, dataset):
		self.epochs = epochs
		self.learn_rate = learn_rate
		self.batch_size = batch_size
		self.dataset = dataset
	
	def train(self):
		tensorboard = TensorBoard(log_dir=f'logs/Resnet50-E{self.epochs}-{time.gmtime(0)}')

		opt = Adam(lr=self.learn_rate, decay=self.learn_rate / self.epochs)
		self.model = Resnet50().build(classes=self.dataset.len_classes)
		self.model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
		
		# train the network
		print("[INFO] training network for {} epochs...".format(self.epochs))
		self.model.fit_generator(self.dataset.generator(self.batch_size),
			steps_per_epoch=self.dataset.len_train // self.batch_size,
			epochs=self.epochs,
			callbacks=[tensorboard])
	
	def evaluate(self):
		data = self.dataset.load_test_data()
		predictions = self.model.predict(self.dataset.load_test_data(), batch_size=self.batch_size)
		print(classification_report(self.dataset.labels,
			predictions.argmax(axis=1), target_names=self.dataset.le.classes_))

	def save_model(self, model_path):
		self.model.save(model_path)