from neuralNetwork import NeuralNetwork
from dataset import dataset_factory
import matplotlib.pyplot as plt
from utils import get_args
import os

args = get_args()

batch_size = args["batch_size"]
dataset = dataset_factory(args["dataset"], (224, 224, 3))

net = NeuralNetwork()
net.train(dataset, args["learn_rate"], args["epochs"], batch_size)
net.evaluate(dataset, batch_size)
net.save_model(args["model"])
dataset.save_labels(args["le"])
net.save_learn_curve(args["plot"])

