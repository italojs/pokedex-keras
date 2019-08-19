from neuralNetwork import NeuralNetwork
from dataset import Dataset
import getArgs
import gc

args = getArgs.train()

batch_size = args["batch_size"]
dataset = Dataset(args["dataset"])
net = NeuralNetwork(args["epochs"], batch_size, args["learn_rate"], dataset)
net.train()
net.evaluate(dataset)
net.save_model(args["model"])
dataset.save_labels(args["le"])
