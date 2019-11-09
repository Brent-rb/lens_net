# import the necessary packages
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.layers.convolutional import Convolution2D
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import random
import pickle
import cv2
import json
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
args = vars(ap.parse_args())


def load_train():
	x_train = []
	y_train = []
	
	for imagePath in sorted(list(paths.list_images(args["dataset"]))):
		number = imagePath.split(".")[0][-1:]
		
		with open(os.path.join(args["dataset"], f"data_{number}.json"), "r") as jsonFile:
			data = json.loads(jsonFile.read())
			image = cv2.imread(imagePath, 1)

			for row in range(10):
				for column in range(10):
					startY = row * 40
					endY = startY + 39
					startX = column * 40
					endX = startX + 39

					subImage = image[startY: endY, startX: endX]
					currentData = data[row][column]
					e1 = currentData['e1']
					e2 = currentData['e2']
					g1 = currentData['g1']
					g2 = currentData['g2']

					subImage = subImage.astype(np.float32)

					x_train.append(subImage.flatten())
					y_train.append([e1, e2, g1, g2])

	print(len(x_train))
	return x_train, y_train

def read_and_normalize_train_data():
	train_data, train_target = load_train()
	train_data = np.array(train_data, dtype=np.float32)
	train_target = np.array(train_target, dtype=np.float32)
 
	train_data /= 255.0

	return train_data, train_target

def create_model():
	nb_filters = 32
	nb_conv = 3

	model = Sequential()
	model.add(Dense(2000, input_dim=39*39*3, init="normal", activation="relu"))
	model.add(Dense(1000, activation="relu"))

	model.add(Dense(500, activation="relu"))

	model.add(Dense(250, activation="relu"))

	model.add(Dense(4, activation="linear"))

	model.compile(loss='mean_squared_error', optimizer=Adadelta(), metrics=["accuracy"])
	return model

def train_model(batch_size = 50, nb_epoch = 20):
	num_samples = 2000
	cv_size = 996

	train_data, train_target = read_and_normalize_train_data()

	X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=cv_size, random_state=56741)

	model = create_model()
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_valid, y_valid) )

	predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
	"""compare = pd.DataFrame(data={'original':y_valid.reshape((cv_size, 4)),
             'prediction':predictions_valid.reshape((cv_size, 4))})
	compare.to_csv('compare.csv')
	"""

	print(predictions_valid)

	return model

train_model(nb_epoch = 200)