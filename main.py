# USAGE
# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background


# import the necessary packages

from neuralNetwork import NeuralNetwork
from dataset import dataset_factory
import matplotlib.pyplot as plt
from utils import get_args
import os

args = get_args()

# initialize the initial learning rate, batch size, and number of
# epochs to train for
BS = args["batch_size"]
EPOCHS = args["epochs"]

dataset = dataset_factory(args["dataset"], (224, 224, 3))

# initialize the optimizer and model
print("[INFO] compiling model...")

net = NeuralNetwork()
net.train(dataset, args["learn_rate"], EPOCHS, BS)
net.evaluate(dataset, BS)
net.save_model(args["model"])
dataset.save_labels(args["le"])
net.save_learn_curve(args["plot"])

