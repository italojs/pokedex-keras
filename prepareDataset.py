import split_folders
from imutils import paths
import cv2

split_folders.ratio('./dataset', output="output", seed=10657, ratio=(.8, .0, .2))

imagePaths = list(paths.list_images('./output'))

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    cv2.imwrite(imagePath, image)