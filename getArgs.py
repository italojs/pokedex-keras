import argparse

def train():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-m", "--model", default="resnet50.model", type=str,
        help="path to trained model")
    ap.add_argument("-l", "--le", default="le.pickle", type=str,
        help="path to label encoder")
    ap.add_argument("-r", "--learn_rate", type=float, default=1e-4,
        help="train learn rate")
    ap.add_argument("-b", "--batch_size", type=float, default=8,
        help="how much images will be handdled by epoch")
    ap.add_argument("-e", "--epochs", type=int, default=50,
        help="how much epochs your n_network will train")
    return vars(ap.parse_args())
    
def predict():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="resnet50.model", type=str,
        help="path to trained model")
    ap.add_argument("-l", "--le", default="le.pickle", type=str,
        help="path to label encoder")
    ap.add_argument("-i", "--image", required=True, type=str,
        help="image to predict")
    return vars(ap.parse_args())