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
	
	def train(self, dataset,learn_rate, epochs, batch_size):
		tensorboard = TensorBoard(log_dir='logs/Resnet50-{}'.format(time()))

		opt = Adam(lr=learn_rate, decay=learn_rate / epochs)
		self.model = Resnet50().build(classes=dataset.len_classes)
		self.model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
		
		# train the network
		print("[INFO] training network for {} epochs...".format(epochs))
		H = self.model.fit_generator(dataset.generator(batch_size),
			steps_per_epoch=dataset.len_train // batch_size,
			epochs=epochs,
			callbacks=[tensorboard])

		self.history = H.history
	
	def evaluate(self, dataset, batch_size):
		predictions = self.model.predict(dataset.len_train, batch_size=batch_size)
		
		print(classification_report(dataset.testY.argmax(axis=1),
			predictions.argmax(axis=1), target_names=dataset.labelEncoder.classes_))

	def save_model(self, model_path, ):
		self.model.save(model_path)