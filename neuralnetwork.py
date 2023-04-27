import math
import pandas as pd
import numpy as np
import pickle
import os

# Neural Network class is used to store a neural network once it has been trained so that users do not have 
# to train multiple times
class neuralNetwork:
    def __init__(self, weights_h, weights_o, bias_h, bias_o):
        self.weights_h = weights_h
        self.weights_o = weights_o
        self.bias_h = bias_h
        self.bias_o = bias_o

    # Save function saves the Neural Networks in binary using pickle
    def save(self, type):
        if not os.path.exists("NeuralNetworks"):
            os.makedirs("NeuralNetworks") 
        with open("NeuralNetworks/weights_h_" + type + ".txt", "wb") as f:
            pickle.dump(self.weights_h, f)
        with open("NeuralNetworks/weights_o_" + type + ".txt", "wb") as f:
            pickle.dump(self.weights_o, f)
        with open("NeuralNetworks/bias_h_" + type + ".txt", "wb") as f:
            pickle.dump(self.bias_h, f)
        with open("NeuralNetworks/bias_o_" + type + ".txt", "wb") as f:
            pickle.dump(self.bias_o, f)
    # Read function can open a saved neural network
    def read(self, type):
        with open("NeuralNetworks/weights_h_" + type + ".txt", "rb") as f:
            self.weights_h = pickle.load(f)
        with open("NeuralNetworks/weights_o_" + type + ".txt", "rb") as f:
            self.weights_o = pickle.load(f)
        with open("NeuralNetworks/bias_h_" + type + ".txt", "rb") as f:
            self.bias_h = pickle.load(f)
        with open("NeuralNetworks/bias_o_" + type + ".txt", "rb") as f:
            self.bias_o = pickle.load(f)


# Activation function
def sigmoid(x):
    # If statement prevents overflow error within the application
    if x < 0:
        return math.pow(math.e, x) / (1 + math.pow(math.e, x))
    else:
        return 1 / (1 + math.pow(math.e, -x))

# The algorithm has slight differences when running with one input architecture vs 
# multiple input architecture. That is why there are seperate functions to train the 
# 01 vs 04 datasets
def train_neural_net_01(df):
    # Current Architecture 784 - 100 - 1

    #Epochs - how many times the algorithm runs for one list of inputs
    # epochs = 100
    
    #Learning Rate
    alpha = 0.5

    # 784 X 100
    weights_h = []

    # 100 X 1
    weights_o = []

    # 100 X 1
    bias_h = []

    # 1 X 1
    bias_o = 0

    # Fill weight lists with values between -1 and 1 randomly
    for a in range(len(df.columns) - 1):
        weights_h.append(list(np.random.uniform(-1,1,100)))
    
    for b in range(100):
        weights_o.append(np.random.uniform(-1,1))
        bias_h.append(np.random.uniform(-1,1))

    bias_o = np.random.uniform(-1,1)

    # Runs through all of the given training data
    for i in range(len(df.index)):
        # Retrieve row from dataframe for input values and label
        input = df.loc[i].tolist()
        # Ouput value (y)
        label = input.pop(0)
        # Normalize input values
        input = np.divide(input, 255)

        # Train neural network epochs number of times for current input data
        # for j in range(epochs):

        # Forward Pass
        # Generate hidden outputs
        h_in = np.add(np.dot(list(np.array(weights_h).transpose()), input), bias_h)
        h_out = list(map(sigmoid, h_in))

        # Generate full output
        o_out = sigmoid(np.add(np.dot(weights_o, h_out), bias_o))
        # Forward pass complete

        # Backward pass
        # Calculate deltas
        delta_o = (label - o_out) * (o_out * (1 - o_out))

        # Propagate Deltas backward
        delta_h = np.multiply(np.dot(delta_o, weights_o), (np.multiply(h_out, np.subtract(1, h_out))))
        
        # Weight updates
        weights_o = np.add(weights_o, (alpha * np.dot(h_out, delta_o)))
        bias_o = np.add(bias_o, alpha * delta_o)
        weights_h = np.add(weights_h, (alpha * np.outer(input, delta_h)))
        bias_h = np.add(bias_h, (alpha * delta_h))

        # Print update on how far along training is
        if i % 100 == 0:
            print(str(i) + " rows trained")
    
    # Creates nerual network object to be passed between functions
    neuralNet = neuralNetwork(weights_h, weights_o, bias_h, bias_o)
    # Saves neural net to text files as binary
    # neuralNet.save("01")
    return neuralNet

# Testing function for the 01 neural network
def test_neural_network_01(df, neuralNetwork):
    
    # How many times the neural net correctly classifies the data
    success = 0
    
    # Weights retrieved from the neural net object
    weights_h = neuralNetwork.weights_h
    weights_o = neuralNetwork.weights_o
    bias_h = neuralNetwork.bias_h
    bias_o = neuralNetwork.bias_o

    # Loops through all of the given test data
    for i in range(len(df.index)):
        # Retrieve row from dataframe for input values and label
        input = df.loc[i].tolist()
        # Ouput value (y)
        label = input.pop(0)
        # Normalize input values
        input = np.divide(input, 255)

        # Forward Pass
        # Generate hidden outputs
        h_in = np.add(np.dot(list(np.array(weights_h).transpose()), input), bias_h)
        h_out = list(map(sigmoid, h_in))

        # Generate full output
        o_out = sigmoid(np.add(np.dot(weights_o, h_out), bias_o))
        # Forward pass complete

        # Classify neural network output
        if o_out > 0.5:
            o_out = 1
        else:
            o_out = 0

        if label == o_out:
            success += 1

    return success / (len(df.index) - 1)

# Training neural network for the 04 dataset
def train_neural_net_04(df):
    # Current Architecture 784 - 100 - 5

    #Epochs - how many times the algorithm runs for one list of inputs
    # epochs = 100
    
    # Learning Rate
    alpha = 0.5

    # 784  X 100
    weights_h = []

    # 100 X 5
    weights_o = []

    # 100 X 1
    bias_h = []

    # 5 X 1
    bias_o = []

    # Fill weight lists with values between -1 and 1 randomly
    for a in range(len(df.columns) - 1):
        weights_h.append(list(np.random.uniform(-1,1,100)))
    
    for b in range(100):
        weights_o.append(np.random.uniform(-1,1,5))
        bias_h.append(np.random.uniform(-1,1))

    for c in range(5):
        bias_o.append(np.random.uniform(-1,1))

    # Runs through all of the given training data
    for i in range(len(df.index)):
        # Retrieve row from dataframe for input values and label
        input = df.loc[i].tolist()
        # Label value in data
        target = input.pop(0)
        # Normalize input values
        input = np.divide(input, 255)
        label = []
        # For dataset with label values 0 - 4 we create a list of length 5 and assign a 1 at 
        # index = label. This is our output value (y)
        for output_vals in range(5):
            if output_vals == target:
                label.append(1)
            else:
                label.append(0)

        # Train neural network epochs number of times for current input data
        # for j in range(epochs):

        # Forward Pass
        # Generate hidden outputs
        h_in = np.add(np.dot(list(np.array(weights_h).transpose()), input), bias_h)
        h_out = list(map(sigmoid, h_in))

        # Generate full output
        o_out = list(map(sigmoid, (np.add(np.dot(list(np.array(weights_o).transpose()), h_out), bias_o))))
        # Forward pass complete

        # Backward pass
        # Calculate deltas
        delta_o = np.multiply(np.subtract(label, o_out), np.multiply(o_out, np.subtract(1, o_out)))

        # Propagate Deltas backward
        delta_h = np.multiply(np.dot(delta_o, np.array(weights_o).transpose()), (np.multiply(h_out, np.subtract(1, h_out))))
        
        # Weight updates
        weights_o = np.add(weights_o, (alpha * np.outer(h_out, delta_o)))
        bias_o = np.add(bias_o, alpha * delta_o)
        weights_h = np.add(weights_h, (alpha * np.outer(input, delta_h)))
        bias_h = np.add(bias_h, (alpha * delta_h))
        
        # Print update on how far along training is
        if i % 100 == 0:
            print(str(i) + " rows trained")
    # Creates nerual network object to be passed between functions
    neuralNet = neuralNetwork(weights_h, weights_o, bias_h, bias_o)
    # Saves neural net to text files as binary
    # neuralNet.save("04")
    return neuralNet

# Testing function for the 04 neural network
def test_neural_network_04(df, neuralNetwork):

    # How many times the neural net correctly classifies the data
    success = 0
    
    # Weights retrieved from the neural net object
    weights_h = neuralNetwork.weights_h
    weights_o = neuralNetwork.weights_o
    bias_h = neuralNetwork.bias_h
    bias_o = neuralNetwork.bias_o

    # Loops through all of the given test data
    for i in range(len(df.index)):
        # Retrieve row from dataframe for input values and label
        input = df.loc[i].tolist()
        # Ouput value (y)
        label = input.pop(0)
        # Normalize input values
        input = np.divide(input, 255)

        #Forward Pass
        # Generate hidden outputs
        h_in = np.add(np.dot(list(np.array(weights_h).transpose()), input), bias_h)
        h_out = list(map(sigmoid, h_in))

        #Generate full output
        o_out = list(map(sigmoid, (np.add(np.dot(list(np.array(weights_o).transpose()), h_out), bias_o))))
        #Forward pass complete

        # Finds the index of the max output value
        out_index, match = -1, 0
        for index in range(len(o_out)):
            if o_out[index] > match:
                match = o_out[index]
                out_index = index

        if out_index == label:
            success += 1

    return success / (len(df.index) - 1)

# Trains a neural network on the mnist_train_0_1.csv file and tests neural network on the mnist_test_0_1.csv file
def run_01():
    # 784 columns (values) per row
    # 12665 rows
    print("Training Started")
    train = pd.read_csv("data/mnist_train_0_1.csv", header=None)
    neuralNet = train_neural_net_01(train)
    print("Testing Started")
    test = pd.read_csv("data/mnist_test_0_1.csv", header=None)
    success = test_neural_network_01(test, neuralNet)
    print("Neural Network Accuracy for mnist_0_1 data: " + str(success))

# Trains a neural network on the mnist_train_0_4.csv file and tests neural network on the mnist_test_0_4.csv file
def run_04():
    # 784 columns (values per row)
    # 30595 rows
    print("Training Started")
    train = pd.read_csv("data/mnist_train_0_4.csv", header=None)
    neuralNet = train_neural_net_04(train)
    print("Testing Started")
    test = pd.read_csv("data/mnist_test_0_4.csv", header=None)
    success = test_neural_network_04(test, neuralNet)
    print("Neural Network Accuracy for mnist_0_4 data: " + str(success))

# Allows user to test neural network accuracy for a given data set.
def test_neural_net_already_created_01():
    nn = neuralNetwork([],[],[],[])
    nn.read("01")
    test = pd.read_csv("data/mnist_test_0_1.csv", header=None)
    success = test_neural_network_01(test, nn)
    print("Neural Network Accuracy for mnist_0_1 data: " + str(success))

def test_neural_net_already_created_04():
    nn = neuralNetwork([],[],[],[])
    nn.read("04")
    test = pd.read_csv("data/mnist_test_0_4.csv", header=None)
    success = test_neural_network_04(test, nn)
    print("Neural Network Accuracy for mnist_0_4 data: " + str(success))

run_04()