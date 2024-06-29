import numpy as np
from scipy import ndimage
import struct
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import math
import pickle
import time

class NeuralNetwork:
    # Define the sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s*(1-s)
    
    def load_network(self):
        file = open ('weights', 'rb')
        self.weights = pickle.load(file)

        file = open('biases', 'rb')
        self.biases = pickle.load(file)
    
    def save_network(self):
        file = open('weights', 'wb')      
        pickle.dump(self.weights, file)

        file = open('biases', 'wb')
        pickle.dump(self.biases, file)
    
    def init_weights_xavier(self):
        weights = []                #xavier something I think
        for i in range(self.calcs): #init sections of weights
            front_layer_count = self.neurons_per_layer[i + 1]
            back_layer_count = self.neurons_per_layer[i]
            w_range = math.pow((6 / (back_layer_count + front_layer_count)), 0.5)
            weights.append(np.random.uniform(-w_range, w_range, (front_layer_count, back_layer_count))) #back neuron w's
        return weights
    
    def init_weights_gaussian(self):
        weights = []                #xavier something I think
        for i in range(self.calcs): #init sections of weights
            front_layer_count = self.neurons_per_layer[i + 1]
            back_layer_count = self.neurons_per_layer[i]
            weights.append(np.random.normal(0, 1/math.sqrt(back_layer_count), (front_layer_count, back_layer_count)))
        return weights

    def __init__(self):
        # Load image data
        with open('samples/train-images.idx3-ubyte', 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            self.data = data.reshape((size, nrows, ncols))

        #load label data
        with open('samples/train-labels.idx1-ubyte', 'rb') as file:
            magic_number = int.from_bytes(file.read(4), 'big')
            num_items = int.from_bytes(file.read(4), 'big')
            self.labels = np.frombuffer(file.read(), dtype=np.uint8)


        # Initialize network structure
        self.input_neurons = self.data[0, :, :].flatten()
        self.neurons_per_layer = [len(self.input_neurons), 16,  16, 10]
        self.num_layers = len(self.neurons_per_layer)
        self.calcs = self.num_layers - 1
        self.last_layer_index = self.num_layers - 1

        # Initialize activations, weights, biases, and errors
        self.activations = [np.zeros(neurons) for neurons in self.neurons_per_layer]

        self.weights = self.init_weights_gaussian()
            
        self.weights_derivatives = []
        for i in range(self.calcs): #init sections of weights derivatives
            front_layer_count = self.neurons_per_layer[i + 1]
            back_layer_count = self.neurons_per_layer[i]
            self.weights_derivatives.append(np.zeros((front_layer_count, back_layer_count), dtype=float)) 

        self.biases = [np.zeros(self.neurons_per_layer[i + 1]) for i in range(self.calcs)]
        self.biases_derivatives = [np.zeros(self.neurons_per_layer[i + 1]) for i in range(self.calcs)]

        self.errors = [] #confusing but trust me it works
        for layer_index in range (self.calcs):
            self.errors.append(np.zeros((self.neurons_per_layer[layer_index+1], self.neurons_per_layer[layer_index])))
        
        self.summed_inputs = [np.zeros(neurons) for neurons in self.neurons_per_layer[1:]] # dont do calc for first layer cuz its an input from the image
        self.expected_values = np.zeros(self.neurons_per_layer[self.last_layer_index], dtype=float)

        image_rots = []
        for image in self.data:
            image_rot = ndimage.rotate(image, 10, reshape=True)
            image_rots.append(image_rot)

        print("done")
        # img = self.data[1, :, :]
        # img_rot = ndimage.rotate(img, 10, reshape=True)
        # plt.imshow(img_rot, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        # img = self.data[1, :, :]
        # img_rot = ndimage.rotate(img, 0, reshape=True)
        # plt.imshow(img_rot, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        # img = self.data[1, :, :]
        # img_rot = ndimage.rotate(img, -10, reshape=True)
        # plt.imshow(img_rot, cmap='gray', vmin=0, vmax=255)
        # plt.show()
    
    def load_input(self, image):
        self.activations[0] = self.data[image, :, :].flatten() / 255.0
    
    def feed_forward(self):
            for layer_index in range(self.calcs):
                self.summed_inputs[layer_index] = np.dot(self.weights[layer_index], self.activations[layer_index]) + self.biases[layer_index]
                self.activations[layer_index + 1] = self.sigmoid(self.summed_inputs[layer_index])

    def quadratic_cost_error(self):
        return (self.activations[self.last_layer_index] - self.expected_values) * self.sigmoid_prime(self.summed_inputs[-1])
    
    def cross_entropy_cost_error(self):
        return (self.activations[self.last_layer_index] - self.expected_values)

    def stochastic(self, epochs, batch_size, learning_rate, regularization):
        # Load input
        
        for epoch in range(epochs):
            image = 0
            while(image < 60000):
                #print(image)
                for counts in range(batch_size):
                    self.load_input(image)
                    self.feed_forward()

                    self.expected_values = np.zeros(self.neurons_per_layer[self.last_layer_index], dtype=float)
                    self.expected_values[self.labels[image]] = 1

                    self.errors[-1] = self.cross_entropy_cost_error()
                    for i in range(len(self.errors) - 2, -1, -1):
                        self.errors[i] = np.dot(np.transpose(self.weights[i+1]),self.errors[i+1]) * self.sigmoid_prime(self.summed_inputs[i])

                    for layer in range(len(self.weights)):
                        self.biases_derivatives[layer] += self.errors[layer]
                        for neuron in range(len(self.weights[layer])):
                                self.weights_derivatives[layer][neuron] += self.activations[layer] * self.errors[layer][neuron]

                    image += 1

                self.biases -= np.multiply(self.biases_derivatives, (learning_rate / batch_size))
                self.weights = np.multiply((self.weights), (1 - ((learning_rate * regularization) / len(self.data)))) - np.multiply((learning_rate / batch_size), self.weights_derivatives)

                #reset derivatives
                self.weights_derivatives = []
                for i in range(self.calcs): #reset weight derivatives to zero
                    front_layer_count = self.neurons_per_layer[i + 1]
                    back_layer_count = self.neurons_per_layer[i]
                    self.weights_derivatives.append(np.zeros((front_layer_count, back_layer_count), dtype=float)) 

                self.biases_derivatives = [np.zeros(self.neurons_per_layer[i + 1]) for i in range(self.calcs)]
        
            print("epoch: " + str(epoch) + " " + self.evaluate())

    def evaluate(self):

        # Load training data
        with open('samples/t10k-images.idx3-ubyte', 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows, ncols))

        #load training label data
        with open('samples/t10k-labels.idx1-ubyte', 'rb') as file:
                magic_number = int.from_bytes(file.read(4), 'big')
                num_items = int.from_bytes(file.read(4), 'big')
                labels = np.frombuffer(file.read(), dtype=np.uint8)

        num_correct = 0
        num_incorrect = 0

        image = 0
        while image < len(data):
            self.load_input(image)
            self.feed_forward()

            if self.activations[self.last_layer_index].argmax() == self.labels[image]:
                num_correct += 1
            else:
                num_incorrect += 1

            image+=1
            
        return ("rate: " + str(num_correct / (num_correct + num_incorrect)))
                    
