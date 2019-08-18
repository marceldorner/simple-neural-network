import math
import numpy

class neuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.learning_rate = learning_rate

        self.w_input_hidden  = numpy.random.normal(0.0, 1.0 / math.sqrt(self.input_nodes), (self.hidden_nodes, self.input_nodes))
        self.w_hidden_output = numpy.random.normal(0.0, 1.0 / math.sqrt(self.hidden_nodes), (self.output_nodes, self.hidden_nodes))
    
        self.sigmoid = lambda x: 1 / (1 + numpy.exp(-x))
        self.sigmoid_derivative = lambda x: x * (1 - x)
    
    def train(self, input_list, target_list):
        # Convert input list and target list to 2D array with 1 column (= vector)
        input_vector = numpy.array(input_list, ndmin=2).T
        target_vector = numpy.array(target_list, ndmin=2).T
        
        hidden_in = numpy.dot(self.w_input_hidden, input_vector)
        hidden_out = self.sigmoid(hidden_in)
        
        output_in = numpy.dot(self.w_hidden_output, hidden_out)
        output_out = self.sigmoid(output_in)
        
        # Calculate error values between hidden layer and output layer
        output_error = target_vector - output_out
        
        # Calculate error valu es between input layer and output layer
        hidden_error = numpy.dot(self.w_hidden_output.T, output_error)

        # Update weights between hidden layer and output layer
        self.w_hidden_output += self.learning_rate * numpy.dot((output_error * self.sigmoid_derivative(output_out)), hidden_out.T)

        # Update weights between input layer and hidden layer
        self.w_input_hidden += self.learning_rate * numpy.dot((hidden_error * self.sigmoid_derivative(hidden_out)), input_vector.T)
    
    def query(self, input_list):
        # Convert input list to 2D array with 1 column (= vector)
        input_vector = numpy.array(input_list, ndmin=2).T
        
        hidden_in = numpy.dot(self.w_input_hidden, input_vector)
        hidden_out = self.sigmoid(hidden_in)
        
        output_in = numpy.dot(self.w_hidden_output, hidden_out)
        output_out = self.sigmoid(output_in)
        
        return output_out

input_nodes  = 784  # Each MNIST image has 28 x 28 = 784 pixels
hidden_nodes = 100  # Number of hidden nodes should be smaller than number of input nodes to "compress" the learned data (see page 148)
output_nodes = 10   # Each output node represents a number between 0 and 9 (output node 0 = number 0, ...)

learning_rate = 0.2 # Step size used during gradient descent (high values may cause repeated overleaping of the error functions minimum)

# Initialize neural network
neuralNetwork = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Train neural network
training_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data = training_file.readlines()
training_file.close()

print("Training neural network with", len(training_data), "records ...")

for record in training_data:
    values = record.split(',')
    
    # Plot pixel data
    # pixel_data = numpy.asfarray(values[1:]).reshape((28, 28))
    # matplotlib.pyplot.imshow(pixel_data, cmap='Greys', interpolation='None')
    # matplotlib.pyplot.pause(0.01)
    
    # Prepare input list (page 142)
    input_list = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
    
    # Prepare target list: First set all values to 0.01, then override target value to 0.99 (page 144)
    # Example for target value = 7:
    # [0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01] --> [0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.99 0.01 0.01]
    target_list = numpy.zeros(output_nodes) + 0.01
    target_list[int(values[0])] = 0.99
    
    neuralNetwork.train(input_list, target_list)

# Test neural network
test_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data = test_file.readlines()
test_file.close()

score = 0

print("Testing neural network with", len(test_data), "records ...")

for record in test_data:
    values = record.split(',')
    
    # Prepare input list (page 142)
    input_list = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
    
    # Query the neural network (perform feedforward and backpropagation)
    output_list = neuralNetwork.query(input_list)
    
    label = numpy.argmax(output_list)
    correct_label = int(values[0])

    if (label == correct_label):
        score += 1

print("Performance =", score / len(test_data))

input("Press ENTER to exit...")
