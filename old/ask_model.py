import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from sklearn.utils import shuffle
import cv2

from keras.models import load_model
from keras.models import model_from_json
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
import json
from sklearn.model_selection import train_test_split
import argparse
from keras.optimizers import SGD, Adam

conversion_min = 0
conversion_max = 0

def load_dataset(directory, filename):
    global conversion_max
    global conversion_min

    inputFile = os.path.join(directory, filename)

    training_data = []
    test_data = []

    with open(inputFile, "r") as jsonFile:
        jsonObject = json.loads(jsonFile.read())

        for entry in jsonObject:
            galName = os.path.join(directory, entry['galaxy'])
            g1 = entry['g1']
            g2 = entry['g2']

            training_data.append(cv2.imread(galName, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            test_data.append([g1, g2])

        training_data = np.array(training_data, dtype=np.float32) / 255.0
        test_data = np.array(test_data, dtype=np.float32)

        conversion_min = test_data.min()
        test_data -= test_data.min()
        conversion_max = test_data.max()
        test_data /= test_data.max()



        x_train, x_test, y_train, y_test = train_test_split(training_data, test_data, test_size=0.25, random_state=56741)

        return x_train, x_test, y_train, y_test


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="The input image to evaluate.")
ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset of images")
ap.add_argument("-j", "--json", required=True, help="Name of the data json file.")
ap.add_argument("-m", "--model", required=True, help="Name of the model file.")
args = vars(ap.parse_args())

image_size = 40
image_depth = 1
num_classes = 2

x_train, x_test, y_train, y_test = load_dataset(args["dataset"], args["json"])
x_train = x_train.reshape((3750, 40, 40, 1))
x_test = x_test.reshape((1250, 40, 40, 1))

# load json and create model
json_file = open(f"{args['model']}.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(f"{args['model']}.h5")
adam = Adam(lr=0.001)
model.compile(optimizer= adam, loss='mean_squared_error', metrics=['accuracy'])
print("Loaded model from disk")

images = [cv2.imread(args["input"], cv2.IMREAD_GRAYSCALE).astype(np.float32)]
images = np.array(images, dtype=np.float32) / 255.0

predictions = model.predict(images.reshape(1, 40, 40, 1))

for prediction in predictions:
    print(f"Prediction G1:{(prediction[0] * conversion_max) + conversion_min} G2:{(prediction[1] * conversion_max) + conversion_min}")