import random
import numpy as np

class qudratic_cost(object):
    # returns outer layer error wrt cost function
    @staticmethod
    def error(outer_activation, y, z):
        return (outer_activation - y) * sigmoid_prime(z)
    
    # returns the cost of output a and desired output y
    @staticmethod
    def cost_return(a, y):
        return 0.5 * np.linalg.norm(a - y)**2

class cross_entropy_cost(object):
    # returns outer layer error wrt cost function
    @staticmethod
    def error(outer_activation, y, z):
        return outer_activation - y
    
    # returns the cost of output a and desired output y
    @staticmethod
    def cost_return(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    

class Network(object):
    def __init__(self, sizes, cost = cross_entropy_cost):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.cost = cost
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]



    # assumes a is nx1 nparray and returns the index with the highest activation from index 0-9, must be grayscaled 0-1
    def run_input(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return np.argmax(a)
    
    """
    training_data must be a list/tuple consisting of tuples (x, y) with x input, y output
    x must be nx1 nparray
    y must be output 1x10 nparray
    epochs is the number of exhaustions for the data set
    eta is the training rate (small delta to push the cost function in the direction of negative gradient)
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda):
        training_data = list(training_data)
        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            #SGD step for minibatch
            for minibatch in mini_batches:
                sum_grad_w = [np.zeros(w.shape) for w in self.weights]
                sum_grad_b = [np.zeros(b.shape) for b in self.biases]
                
                for x, y in minibatch:
                    grad_w, grad_b = self.backprop(x, y)
                    sum_grad_w = [sgw + gw for sgw, gw in zip(sum_grad_w, grad_w)]
                    sum_grad_b = [sgb + gb for sgb, gb in zip(sum_grad_b, grad_b)]

                # subtracting step from weights and biases and utilizes L2 regularization
                self.weights = [(1 - eta * lmbda / n) * w - eta / mini_batch_size * sgw for w, sgw in zip(self.weights, sum_grad_w)]
                self.biases = [b - eta / mini_batch_size * sgb for b, sgb in zip(self.biases, sum_grad_b)]

            print("Epoch {} complete".format(i))

    def backprop(self, x, y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs = []

        # feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # calculate error z^L with cost function
        z_l = self.cost.error(activations[-1], y, zs[-1])
        grad_w[-1] = np.dot(z_l, np.transpose(activations[-2]))
        grad_b[-1] = z_l

        # calculate error error z^l using error z^(l+1)
        for layer in range(2, self.num_layers):
            z_l = np.dot(np.transpose(self.weights[-layer+1]), z_l) * sigmoid_prime(zs[-layer])
            grad_w[-layer] = np.dot(z_l, np.transpose(activations[-layer-1]))
            grad_b[-layer] = z_l

        return (grad_w, grad_b)
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
