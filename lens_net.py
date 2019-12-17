import os
import cv2
import json
import argparse
import configparser
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

from keras import applications 
from keras import optimizers

from keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential,Model, load_model, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files   
from sklearn.utils import shuffle

from urllib.request import urlopen,urlretrieve
from PIL import Image
from glob import glob

def main():
    global args
    global config

    parse_arguments()
    parse_config()
    overwrite_config()

    # Extract arguments
    shouldTrain = args["train"]
    shouldEvaluate = args["evaluate"]
    shouldPredict = args["predict"]
    shouldRemove = args["remove"]
    inputFilename = args["input"]
    modelFilename = args["model"]
    _, modelName = os.path.split(modelFilename)

    # Safety check
    if (shouldTrain or shouldEvaluate) and shouldPredict:
        print("Error: Cannot train/evaluate and predict at the same time")
        exit()

    if (modelName.lower() == "default"):
        print("Error: Model name may not be 'default'")
        exit()    

    # Actual main function
    # run_lens_net(shouldTrain, shouldEvaluate, shouldPredict, shouldRemove, inputFilename, modelFilename)

def run_lens_net(shouldTrain, shouldEvaluate, shouldPredict, shouldRemove, inputFilename, modelFilename):
    """Creates or loads a neural network model and then trains, evaluates or predicts with the model depending on the input and flags.
    
    Arguments:
        shouldTrain {bool} -- Indicates wether or not we should train the neural network model.
        shouldEvaluate {bool} -- Indicates wether or not we should evaluate the neural network model.
        shouldPredict {bool} -- Indicates wether or not we should use the neural network model to predict the outcome of the input.
        shouldRemove {bool} -- Indicates wether or not we should delete the existing model and create a new one.
        inputFilename {str} -- The filename of the input. Must be a .json when training or evaluating. Can be a .json file or image file when predicting.
        modelFilename {str} -- The filename of the model without extension. The extensions .json and .h5 will be automatically added.
    """

    global config
    # Split the modelFilename in directory and modelName 
    _, modelName = os.path.split(modelFilename)

    # Load or create model
    model = get_model(shouldRemove, modelFilename)
    model.summary()

    try:
        if shouldTrain or shouldEvaluate:
            train_or_evaluate(model, shouldTrain, shouldEvaluate, inputFilename, modelName)
            
        elif shouldPredict:
            predict(model, inputFilename, modelName)

        save_model(model, modelFilename)
        save_config()

    # We might interrupt learning, even on interrupt save model and config
    except KeyboardInterrupt:
        save_model(model, modelFilename)
        save_config()

def train_or_evaluate(model, shouldTrain, shouldEvaluate, inputFilename, modelName):
    """Trains or evaluates the given model with data from inputFilename.
    
    Arguments:
        model {} -- The model to train or evaluate.
        shouldTrain {bool} -- Wether or not we should train the model.
        shouldEvaluate {bool} -- Wether or not we should evaluate the model.
        inputFilename {str} -- The filename of the data object we will use to evaluate or train the model.
        modelName {str} -- The name of the model.
    """

    global config
    image_height = get_image_height()
    image_width = get_image_width()
    image_channels = get_image_depth()

    # Load the training data and reshape it to fit the input layer
    x_train, x_test, y_train, y_test, min_output, max_output = load_dataset(inputFilename)
    # Shape should be (sample_amount, image_height, image_width, image_depth) for TensorFlow
    # Other frameworks use other order
    x_train = x_train.reshape((x_train.shape[0], image_height, image_width, image_channels))
    x_test = x_test.reshape((x_test.shape[0], image_height, image_width, image_channels))

    if shouldTrain:
        print("Training model...")
        # Because we normalize the data we need to save these parameters in order to reconvert the output of the neural network when predicting.
        config[modelName] = {}
        config[modelName]["min_output"] = min_output.item()
        config[modelName]["max_output"] = max_output.item()

        # Train the neural network
        model.fit(x_train, y_train, epochs = config["epochs"], batch_size = config["batch_size"], shuffle=True)
        print("Training complete!")
    
    if shouldEvaluate:
        # Evaluate the neural network
        preds = model.evaluate(x_test, y_test)
        print ("Loss = " + str(preds[0]))
        print ("Test Accuracy = " + str(preds[1]))

        # Print out a summary of the model
        # model.summary()

def predict(model, inputFilename, modelName):
    """Uses the given model to predict the data for the given inputFilename. If the input is a .json data file, all entries will be predicted and we will compare answers. If it's an image file, we will just predict.
    
    Arguments:
        model {} -- The neural network we use to predict.
        inputFilename {str} -- Name of the input file that we will predict. Can be a .json data file or an image file.
        modelName {str} -- Name of the model, used to load the conversion parameters.
    """

    if inputFilename.endswith(".json"):
        predict_json(model, inputFilename, modelName)
    else:
        predict_image(model, inputFilename, modelName)

def predict_json(model, inputFilename, modelName):
    """Uses the given model to make a prediction about all the entries in a data .json file.
    
    Arguments:
        model {} -- The neural network model we will use to predict.
        inputFilename {str} -- The filename of the input file. Must be a .json file.
        modelName {str} -- Name of the modelm used to load the conversion parameters.
    """

    global config
    image_height = get_image_height()
    image_width = get_image_width()
    image_channels = get_image_depth()
    min_output = config[modelName]["min_output"]
    max_output = config[modelName]["max_output"]

    # Load all the input images and normalize them
    predict_data, predict_answers = read_data(inputFilename)
    predict_data /= 255.0

    # Make a prediction about all these images
    predictions = model.predict(predict_data.reshape(len(predict_data), image_height, image_width, image_channels))
    # For each prediction compare it to the actual value
    for i in range(len(predictions)):
        prediction = predictions[i]
        expected = predict_answers[i]

        prediction *= max_output
        prediction += min_output

        print(f"Expected: {expected}, Result: {prediction}")

def predict_image(model, inputFilename, modelName):
    """Uses the given model to make a prediction about an imput image file.
    
    Arguments:
        model {} -- The neural network model we will use to make a prediction.
        inputFilename {str} -- The filename of the image that we will make a prediction about.
        modelName {str} -- The name of the model to retrieve the conversion parameters.
    """

    global config
    image_height = get_image_height()
    image_width = get_image_width()
    image_channels = get_image_depth()
    min_output = config[modelName]["min_output"]
    max_output = config[modelName]["max_output"]

    # Read in the image and normalize it
    predict_data = [cv2.imread(inputFilename, cv2.IMREAD_GRAYSCALE if image_channels == 1 else cv2.IMREAD_COLOR)]
    predict_data /= 255.0

    # Make a prediction 
    predictions = model.predict(predict_data.reshape(len(predict_data), image_height, image_width, image_channels))
    for i in range(len(predictions)):
        prediction = prediction[i]
        prediction *= max_output
        prediction += min_output

        print(f"Prediction result: {prediction}")

def parse_arguments():
    """Parses all the arguments used when running this script.
    """

    global args

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-t", "--train", action="store_true", required=False, help="Use this flag to train the network.")
    ap.add_argument("-e", "--evaluate", action="store_true", required=False, help="Use this flag to evaluate the network, if -t is used this will happen after training.")
    ap.add_argument("-p", "--predict", action="store_true", required=False, help="Use this flag to make a prediction.")
    ap.add_argument("-i", "--input", type=str, required=True, help="The input file, if training this file has to be a .json data file, if predicting this can be either an input file or .json data file.")
    ap.add_argument("-m", "--model", type=str, required=True, help="Name of the model file.")
    ap.add_argument("-r", "--remove", action="store_true", default=False, required=False, help="Use this flag to remove the existing model and create a new one.")
    ap.add_argument("--normalize", action="store_true", required=False, help="Use this flag to normalize the output data.")
    ap.add_argument("--mode", type=str, required=False, help="Overwrites the default mode that the image will be read in.")
    ap.add_argument("--rate", type=float, required=False, help="Overwrites the default learning rate.")
    ap.add_argument("--classes", type=str, required=False, help="Overwrites the default output classes.")
    ap.add_argument("--images", type=str, required=False, help="Overwrites the default images we will use as input.")
    ap.add_argument("--batch_size", type=int, required=False, help="Overwrites the default batch size.")
    ap.add_argument("--epochs", type=int, required=False, help="Overwrites the default epochs.")
    ap.add_argument("--loss_function", type=str, required=False, help="Overwrites the default loss funciton.")
    args = vars(ap.parse_args())

def parse_config():
    """Loads a config file if it exists else it creates a config object.
    """

    global config

    if(os.path.isfile("default_config.json")):
        read_config()
    else:
        create_config()

def read_config():
    """Reads a config file from disk. If this fails a config object will be created.
    """

    global config

    try:
        with open("config.json", "r") as configFile:
            config = json.loads(configFile.read())
    except:
        create_config()

def save_config():
    """Saves the config object to disk.
    """

    global config

    with open("config.json", "w") as configFile:
        configFile.write(json.dumps(config, indent=4, sort_keys=True))

    print("Config saved.")

def create_config():
    """Creates a default config object.
    """

    global config

    config = {}
    config = {
        "image_width": 40,
        "image_height": 40,
        "test_ratio": 0.25,
        "classes": [
            "g1",
            "g2"
        ],
        "images": [
            "galaxy"
        ],
        "mode": "rgb",
        "random_state": 56741,
        "learning_rate": 0.001,
        "loss_function": "mean_absolute_error",
        "metrics": [
            "mean_absolute_error",
            "accuracy"
        ],
        "batch_size": 64,
        "epochs": 50,
        "normalize_output": False
    }

"""ap.add_argument("--normalize", type=bool, action="store_true", required=False, help="Use this flag to normalize the output data.")
    ap.add_argument("--mode", type=str, required=False, default="", required=False, help="Overwrites the default mode that the image will be read in.")
    ap.add_argument("--rate", type=int, required=False, default=0, required=False, help="Overwrites the default learning rate.")
    ap.add_argument("--classes", type=str, required=False, default="", help="Overwrites the default output classes.")
    ap.add_argument("--images", type=str, required=False, default="", help="Overwrites the default images we will use as input.")
    ap.add_argument("--batchsize", type=int, required=False, default=0, help="Overwrites the default batch size.")
    ap.add_argument("--epochs", type=int, required=False, default=0, help="Overwrites the default epochs.")
    ap.add_argument("--lossfunction", type=str, required=False, default="", help="Overwrites the default loss funciton.")
    

Returns:
    [type] -- [description]
"""
def overwrite_config():
    global config
    global args

    if(args["normalize"] != None):
        config["normalize_output"] = args["normalize"]

    if(args["mode"] != None):
        config["mode"] = args["mode"]

    if(args["rate"] != None):
        config["learning_rate"] = args["rate"]

    if(args["classes"] != None):
        config["classes"] = [x.strip() for x in args["classes"].split(',')]

    if(args["images"] != None):
        config["images"] = [x.strip() for x in args["images"].split(',')]

    if(args["batch_size"] != None):
        config["bitch_size"] = args["batch_size"]

    if(args["epochs"] != None):
        config["epochs"] = args["epochs"]

    if(args["loss_function"] != None):
        config["loss_function"] = args["loss_function"]

    print(config)

def read_data(dataFilename):
    """Uses a data.json file to read in all the images listed, alongside the wanted data.
    
    Arguments:
        dataFilename {str} -- Filename of the data.json file.
    
    Returns:
        ([np.array], [np.array]) -- A numpy array of all the images and a numpy array of all the answers.
    """

    global config
    image_color = cv2.IMREAD_GRAYSCALE if config["mode"] == "grayscale" else cv2.IMREAD_COLOR

    # Split the path from the filename 
    directory, _ = os.path.split(dataFilename)    

    # Places to store training data and wanted results
    training_data = []
    result_data = []

    # Try to open the data file
    with open(dataFilename, "r") as jsonFile:
        # It's a JSON array
        jsonObject = json.loads(jsonFile.read())
        
        # Loop over every entry
        for entry in jsonObject:
            result = []
            # Get the wanted results
            for result_class in config["classes"]:
                result.append(entry[result_class])

            first = True
            for imageKey in config["images"]:
                # Galaxy image filename
                imageName = os.path.join(directory, entry[imageKey])
                image = cv2.imread(imageName, image_color).astype(np.float32)

                if first:
                    combinedImage = image
                    first = False
                else:
                    combinedImage = np.concatenate((combinedImage, image), axis = 1) # Stack images horizontal
            
            # Append to our lists
            training_data.append(combinedImage)

            result_data.append(result)

        # Convert to numpy arrays
        training_data = np.array(training_data, dtype=np.float32) 
        result_data = np.array(result_data, dtype=np.float32)

        return training_data, result_data

def load_dataset(dataFilename):
    """Reads in all the data mentioned in the data file and formats this data to be used for training and evaluating.
    
    Arguments:
        dataFilename {str} -- The filename of the data.json file.
    
    Returns:
        (np.array, np.array, np.array, np.array, np.float32, np.float32]) -- training data, test data, training answers, test answers, minimum value of answers, maximum value of answers
    """

    global config
    normalizeOutput = config["normalize_output"]

    # Places to store training data and wanted results
    training_data, result_data = read_data(dataFilename)

    # Normalize training data
    training_data /= 255.0

    # Normalize test data
    if normalizeOutput:
        min_output = np.float32(-1)
        result_data += 1.0
        max_output = np.float32(1) 
    else:
        min_output = np.float32(0)
        max_output = np.float32(1)

    print(f"Training output min, max: {result_data.min()}, {result_data.max()}")

    # Split data in training and test data
    x_train, x_test, y_train, y_test = train_test_split(training_data, result_data, test_size=config["test_ratio"], random_state=config["random_state"])

    return x_train, x_test, y_train, y_test, min_output, max_output

def get_model(shouldRemove, modelFilename):
    """Checks if the model is saved on disk and loads it. If there is no model on disk, create it.
    
    Arguments:
        shouldRemove {bool} -- Wether or not the model on disk should be removed (and if a new model will be created)
        modelFilename {str} -- The filename of the model without extensions.
    
    Returns:
        [type] -- The neural network model.
    """

    # The .json file stores the neural network
    # The h5 file stores the network weights
    jsonName = f"{modelFilename}.json"
    h5Name = f"{modelFilename}.h5"

    # Check if the model exists on disk
    if os.path.isfile(jsonName):
        # Remove if asked, and create a new model
        if(shouldRemove):
            os.remove(jsonName)
            os.remove(h5Name)
            model = create_model_vgg()
        # If we should not remove, read in the model
        else:
            model = read_model(jsonName, h5Name)
    # No model found on disk so create one.
    else:
        model = create_model_vgg()

    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(config["learning_rate"])
    model.compile(optimizer= adam, loss=config["loss_function"], metrics=config["metrics"])

    return model 

def create_model_from_base(base):
    global config
    image_height = get_image_height()
    image_width = get_image_width()
    image_channels = get_image_depth()
    num_classes = len(config["classes"])

    print(f"Creating network with input shape: {image_height}, {image_width}, {image_channels} and output shape: {num_classes}")
    
    # Base our model on 
    base_model = base(include_top=False, input_shape=(image_height, image_width, image_channels))

    flat = Flatten()(base_model.output)
    classLayer = Dense(512, activation="relu")(flat)
    classLayer = Dense(512, activation="relu")(classLayer)
    output = Dense(num_classes, activation="linear")(classLayer)

    model = Model(inputs = base_model.input, outputs = output)

    return model

def create_model_vgg():
    """Creates a model based on the information in the config file
    
    Returns:
        [type] -- The neural network model
    """

    model = create_model_from_base(VGG19)

    print("Model created.")
    return model

def create_model_resnet50():
    """Creates a model based on the information in the config file
    
    Returns:
        [type] -- The neural network model
    """

    """
    # Base our model on 
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(image_height, image_width, image_channels))

    outputLayer = base_model.output
    outputLayer = GlobalAveragePooling2D()(outputLayer)
    outputLayer = Dropout(.1)(outputLayer)

    predictions = Dense(num_classes, activation="linear")(outputLayer)

    model = Model(inputs = base_model.input, outputs = predictions)
    """
    model = create_model_from_base(applications.resnet50.ResNet50)

    print("Model created.")
    return model

def read_model(jsonName, h5Name):
    global config

    # load json and create model
    json_file = open(jsonName, 'r')
    json_data = json_file.read()
    json_file.close()

    model = model_from_json(json_data)

    # load weights into new model
    model.load_weights(h5Name)

    print(f"Loaded model from disk: {jsonName}")
    return model

def save_model(model, modelName):
    directory, name = os.path.split(modelName)
    if not os.path.isdir(directory):
        mkdir(directory)

    # serialize model to JSON
    model_json = model.to_json()
    with open(f"{modelName}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{modelName}.h5")
    print("Model saved to disk.")

def mkdir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

def get_image_width(): 
    global config
    return config["image_width"] * len(config["images"])

def get_image_height():
    global config
    return config["image_height"]

def get_image_depth():
    global config

    if config["mode"] == "rgb":
        return 3
    elif config["mode"] == "grayscale":
        return 1

    return -1

if __name__ == "__main__":
    main()