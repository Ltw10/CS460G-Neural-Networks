# CS460G-Neural-Networks

Machine Learning Neural Network Program for my senior year CS460G course.

No neural network libraries were used. Neural network implemented from scratch.

# Repository Information 

All of the code is inside of the neuralnetwork.py file.

To create a neural network, train, and test on the 01 data set you can run the run_01() function.
To create a neural network, train, and test on the 04 data set you can run the run_04() function.
These functions will display the accuracy of the created neural network on the test data set at the end of runtime.
Usually these functions would save the neural networks created, but that line of code is commented out so that the
neural networks that I created using epochs that have higher accuracy can be tested.

Also included is a folder named NeuralNetworks. Within this folder there are text files which house the neural networks
in binary form. To read these neural networks in and test on them you can use one of the test_neural_net_already_created_01() or 
test_neural_net_already_created_04() functions. These functions help to use already created neural networks.
If the neural networks have not been created or do not exist within the folder then this function will throw an error.
Currently within this folder are the neural networks I trained using epochs which perform slightly better than the neural
networks generated without epochs.

Accuracy output from the saved neural networks can be found below:

Neural Network Accuracy for mnist_0_1 data: 1.0
Neural Network Accuracy for mnist_0_4 data: 0.984624367458155
