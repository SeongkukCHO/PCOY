import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

errors_plt=[]
directory = os.path.dirname(os.path.abspath(__file__))
data_path = directory + "/"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
print("< Completed loading file >")

train_image = np.asfarray(train_data[:, 1:])* (1/255.0) #scale값 축소
train_label = np.asfarray(train_data[:, :1])
test_image = np.asfarray(test_data[:, 1:])* (1/255.0) #scale값 축소
test_label = np.asfarray(test_data[:, :1])

class NeuralNetwork:
    
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                        self.no_of_in_nodes))
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden2 = X.rvs((self.no_of_hidden_nodes, 
                                         self.no_of_hidden_nodes))
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out  = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))
    
    def train(self, input_vector, target_vector):        
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_hidden1 = activation_function(output_vector1)
        
        output_vector1_2 = np.dot(self.weights_in_hidden2, output_hidden1)
        output_hidden2 = activation_function(output_vector1_2)
        
        output_vector2 = np.dot(self.weights_hidden_out , output_hidden2)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_network
        ################ propagation 완료
        # update the weights:
        errors_plt.append(np.mean(abs(output_errors))) 
        
        tmp = output_errors * output_network * (1.0 - output_network)     
        tmp = self.learning_rate  * np.dot(tmp, output_hidden2.T)
        self.weights_hidden_out  += tmp
        # calculate hidden errors:
        hidden_errors1 = np.dot(self.weights_hidden_out.T, 
                               output_errors * output_network * (1.0 - output_network))
        # update the weights:
        tmp = hidden_errors1 * output_hidden2 * (1.0 - output_hidden2)
        self.weights_in_hidden2 += self.learning_rate * np.dot(tmp, output_hidden1.T)
        
        hidden_errors2 = np.dot(self.weights_in_hidden2.T,
                                hidden_errors1 * output_hidden2 * (1-output_hidden2))
        tmp = hidden_errors2 * output_hidden1 * (1-output_hidden1)
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T)
        
        #################### weights_in_hidden1에 대한 backpropagation만 하면 2layer 완료
        
    def mul_train(self, train_data, labels_one_hot_array, iterations):
        w = []
        for iteration in range(iterations):  
            print("Learning status : ", iteration / (iterations) * 100, "%")
            for i in range(len(train_data)):
                self.train(train_data[i], labels_one_hot_array[i])
            w.append((self.weights_in_hidden, self.weights_in_hidden2, self.weights_hidden_out))
            print("Acc = ",(1-errors_plt[len(errors_plt)-1])*100, "%")
            #plt.plot(errors_plt)
            #plt.show()
            errors_plt.clear()
        return w[len(w)-1]
            
    def run(self, input_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = activation_function(np.dot(self.weights_in_hidden, input_vector))
        output_vector = activation_function(np.dot(self.weights_in_hidden2, output_vector))
        output_vector = activation_function(np.dot(self.weights_hidden_out, output_vector))
        return output_vector

NeuralNetwork_of_MNIST = NeuralNetwork(no_of_in_nodes = 784, 
                                       no_of_out_nodes = 10, 
                                       no_of_hidden_nodes = 100,
                                       learning_rate = 0.05)

output_of_MNIST = np.arange(10)
one_hot_label_train = (output_of_MNIST==train_label).astype(np.float32)

FinalWeight = NeuralNetwork_of_MNIST.mul_train(train_image, 
                                               one_hot_label_train, 
                                               iterations=10)

NeuralNetwork_of_MNIST.weights_in_hidden = FinalWeight[0]
NeuralNetwork_of_MNIST.weights_in_hidden2 = FinalWeight[1]
NeuralNetwork_of_MNIST.weights_hidden_out = FinalWeight[2]

errors=0
for i in range(len(train_image)):
    if NeuralNetwork_of_MNIST.run(train_image[i]).argmax() != train_label[i] :
        errors+=1
print("Acc of train : ", ((len(train_image) - errors) / ( len(train_image))) * 100, "%")

errors=0
for i in range(len(test_image)):
    if NeuralNetwork_of_MNIST.run(test_image[i]).argmax() != test_label[i] :
        errors+=1
print("Acc of test : ", ((len(test_image) - errors) / ( len(test_image))) * 100, "%")
