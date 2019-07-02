from neuralNetwork import NeuralNetwork
from dataset import dataset_factory
import getArgs
import gc

args = getArgs.train()

batch_size = args["batch_size"]
dataset = dataset_factory(args["dataset"], (224, 224, 3))
gc.collect()
net = NeuralNetwork()
net.train(dataset, args["learn_rate"], args["epochs"], batch_size)
net.evaluate(dataset, batch_size)
net.save_model(args["model"])
dataset.save_labels(args["le"])
