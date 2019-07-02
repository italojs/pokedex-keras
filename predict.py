from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import getArgs
import pickle
import random
import cv2

args = getArgs.predict()

model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

img = cv2.imread(args["image"])
pokemon = cv2.resize(img, (224, 224))
pokemon = pokemon.astype("float") / 255.0
pokemon = img_to_array(pokemon)
pokemon = np.expand_dims(pokemon, axis=0)

preds = model.predict(pokemon)[0]
label = le.classes_[np.argmax(preds)]

print(label)
cv2.imshow(pokemon)