# import the necessary packages
from sklearn.metrics import classification_report
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from model import Resnet50
import numpy as np
import matplotlib

matplotlib.use("Agg")

class NeuralNetwork:
	
	def train(self, dataset,learn_rate, epochs, batch_size):
		self.epochs = epochs
		opt = Adam(lr=learn_rate, decay=learn_rate / epochs)

		self.model = Resnet50().build(classes=len(dataset.labelEncoder.classes_))

		self.model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
		
		# train the network
		print("[INFO] training network for {} epochs...".format(epochs))
		H = self.model.fit_generator(dataset.generator(batch_size),
			validation_data=(dataset.testX, dataset.testY), 
			steps_per_epoch=len(dataset.trainX) // batch_size,
			epochs=epochs)

		self.history = H.history
	
	def evaluate(self, dataset, batch_size):
		# evaluate the network
		print("[INFO] evaluating network...")
		predictions = self.model.predict(dataset.testX, batch_size=batch_size)
		print(classification_report(dataset.testY.argmax(axis=1),
			predictions.argmax(axis=1), target_names=dataset.labelEncoder.classes_))

	def save_model(self, model_path, ):
		# save the network to disk
		print("[INFO] serializing network to '{}'...".format(model_path))
		self.model.save(model_path)

	def save_learn_curve(self, output):
		# plot the training loss and accuracy
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, self.epochs), self.history["loss"], label="train_loss")
		plt.plot(np.arange(0, self.epochs), self.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, self.epochs), self.history["acc"], label="train_acc")
		plt.plot(np.arange(0, self.epochs), self.history["val_acc"], label="val_acc")
		plt.title("Training Loss and Accuracy on Dataset")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig(output)
