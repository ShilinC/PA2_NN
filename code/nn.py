#!/usr/bin/python 

import numpy as np
import matplotlib.pyplot as plt
from random import *

import sys
sys.path.append('./python-mnist/')
from mnist import MNIST

def ReLU(x):
    return np.maximum(x, 0)

def trick_sigmoid(x):
    return 1.7159*np.tanh(2*x/3)

def trick_dsigmoid(x):
    return 1.7159*(1-np.power(np.tanh(2*x/3),2))*2/3

def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1.0)

def dReLU(x):
    x[x > 0] = 1.0
    x[x <= 0] = 0
    return x

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    # Find the largest a, and subtract it from each a in order to prevent overflow
    x_max = np.max(x, 1).reshape(x.shape[0],1)
    sum_exp_x = np.sum(np.exp(x - x_max),1).reshape(x.shape[0],1) 
    pred_y = np.exp(x - x_max) / (sum_exp_x+0.0) 
    return pred_y

def random_init_weights(input_size, output_size):
    return 0.01 * np.random.randn(input_size, output_size)

def zero_init_bias(output_size):
    return  np.zeros((1, output_size))

class Network():

    def __init__(self, layers, init_method_weights = random_init_weights, init_method_bias = zero_init_bias, activation_fn = "ReLU", \
        learning_rate = 0.01, momentum = 1, epoches = 10, batch_size = 128, dropout_rate = 0.5):
        self.layers = layers
        self.init_method_weights = init_method_weights
        self.init_method_bias = init_method_bias

        self.setup_layers()
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if activation_fn == "sigmoid":
            self.activation_fn = sigmoid
            self.activation_dfn = dsigmoid
        elif activation_fn == "ReLU":
            self.activation_fn = ReLU
            self.activation_dfn = dReLU
        elif activation_fn == "trick_sigmoid":
            self.activation_fn = trick_sigmoid
            self.activation_dfn = trick_dsigmoid

    def setup_layers(self):
        self.w = [ self.init_method_weights(input_size, output_size) for input_size, output_size in zip(self.layers[:-1], self.layers[1:])]
        self.b = [ self.init_method_bias(output_size) for output_size in self.layers[1:]]

    def forward(self, x):
        for weight, bias in zip(self.w[:-1], self.b[:-1]):
            x = self.activation_fn(np.matmul(x, weight) + bias)
        pred_y = softmax(np.matmul(x, self.w[-1]) + self.b[-1])

        return pred_y

    def get_activations(self, x):
        activation = x
        activations = [activation] 
        pre_activations = []

        for weight, bias in zip(self.w[:-1], self.b[:-1]):
            pre_activation = np.matmul(activation, weight) + bias
            pre_activations.append(pre_activation)
            activation = self.activation_fn(pre_activation)
            activations.append(activation)

        pre_activation = np.matmul(activation, self.w[-1]) + self.b[-1]    
        pre_activations.append(pre_activation)    
        activation = softmax(pre_activation)
        activations.append(activation)

        return activations, pre_activations

    def update_mini_batch(self, train_data_batch, train_label_batch):
        dw = [np.zeros(weight.shape) for weight in self.w]
        db = [np.zeros(bias.shape) for bias in self.b]

        for train_data, train_label in zip(train_data_batch, train_label_batch):
            dw_, db_ = self.backpropagation(train_data, train_label)
            dw = [dweight + dweight_ for dweight, dweight_ in zip(dw, dw_)]
            db = [dbias + dbias_ for dbias, dbias_ in zip(db, db_)]

        self.w = [weight + self.learning_rate * dw_ / (train_data_batch.shape[0]+0.0) for weight, dw_ in zip(self.w, dw)]
        self.b = [bias + self.learning_rate * db_ / (train_data_batch.shape[0]+0.0)  for bias, db_ in zip(self.b, db)]

    def backpropagation(self, train_data, train_label):
        train_data = train_data.reshape(1, train_data.shape[0])

        dw = [np.zeros(weight.shape) for weight in self.w]
        db = [np.zeros(bias.shape) for bias in self.b]

        activations, pre_activations = self.get_activations(train_data)
        delta = train_label - activations[-1]

        dw[-1] = np.matmul( activations[-2].transpose(), delta)
        db[-1] = delta

        for idx in range(2, len(self.layers)):
            pre_activation = pre_activations[-idx]
            activation = activations[-idx-1]
            delta = self.activation_dfn(pre_activation) * np.dot(delta, self.w[-idx+1].transpose())
            dw[-idx] = np.dot( activation.transpose(), delta)
            db[-idx] = delta  

        return dw, db

    def loss(self, input_data, one_hot_labels):
        pred_y = self.forward(input_data)
        pred_y[pred_y == 0.0] = 1e-15
        log_pred_y = np.log(pred_y)
        loss_ = -np.sum(one_hot_labels * log_pred_y) / (one_hot_labels.shape[0]+0.0)

        return loss_
 
    def accuracy(self, input_data, labels):
        pred_y = self.forward(input_data)
        pred_class = np.argmax(pred_y, axis=1)
        accuracy_ = np.sum(pred_class == labels) / (pred_class.shape[0]+0.0)

        return accuracy_

    def train(self, training_images, one_hot_train_labels, training_labels, test_images, one_hot_test_labels, test_labels, validation_images, validation_labels, one_hot_validation_labels):

        self.accuracy(training_images, training_labels)
        pred_y = self.forward(training_images)

        batch_count = training_images.shape[0] / self.batch_size
        self.validation_loss = float("inf")
        self.best_validation_weights =[ np.zeros(weight.shape) for weight in self.w]
        self.best_validation_biases = [ np.zeros(bias.shape) for bias in self.b]

        training_accuracy_all = []
        test_accuracy_all = []
        validation_accuracy_all = []

        training_loss_all = []
        test_loss_all = []
        validation_loss_all = []

        for epoch in range(self.epoches):
            idxs = np.random.permutation(training_images.shape[0]) 
            X_random = training_images[idxs]
            Y_random = one_hot_train_labels[idxs]

            for i in range(batch_count):
                train_data_batch = X_random[i * self.batch_size: (i+1) * self.batch_size, :]
                train_label_batch = Y_random[i * self.batch_size: (i+1) * self.batch_size, :]                
                self.update_mini_batch(train_data_batch, train_label_batch)

                loss_ = self.loss(validation_images, one_hot_validation_labels)
                if loss_ <= self.validation_loss:
                    self.validation_loss = loss_
                    self.best_validation_weights = [weight for weight in self.w]
                    self.best_validation_biases = [bias for bias in self.b]
                else:
                    break

                training_accuracy_all.append(self.accuracy(training_images, training_labels))
                test_accuracy_all.append(self.accuracy(test_images, test_labels))
                validation_accuracy_all.append(self.accuracy(validation_images, validation_labels))

                training_loss_all.append(self.loss(training_images, one_hot_train_labels))
                test_loss_all.append(self.loss(test_images, one_hot_test_labels))
                validation_loss_all.append(self.loss(validation_images, one_hot_validation_labels))


        fig1 = plt.figure(1)
        plt.plot(x_, training_accuracy_all,'ro-')
        plt.plot(x_, test_accuracy_all, 'bo-')
        plt.plot(x_, validation_accuracy_all, 'go-')

        plt.legend(['train accuracy', 'test accuracy', 'validation accuracy'], loc='lower right')
        plt.xlabel('Batches', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.title('Accuracy VS Batches', fontsize=15)
        fig1.show()

        fig2 = plt.figure(2)
        plt.plot(x_, training_loss_all,'ro-')
        plt.plot(x_, test_loss_all, 'bo-')
        plt.plot(x_, validation_loss_all, 'go-')

        plt.legend(['train loss', 'test loss', 'validation loss_'], loc='lower right')
        plt.xlabel('Batches', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.title('Loss VS Batches', fontsize=15)
        fig2.show()           


if __name__ == '__main__':
    # Read datasets
    data = MNIST('./python-mnist/data')
    training_images, training_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    training_images = np.array(training_images)
    test_images = np.array(test_images)
    training_labels = np.array(training_labels)
    test_labels = np.array(test_labels)

    training_images = training_images / 127.5 - 1
    test_images = test_images / 127.5 - 1

    classes = 10
    one_hot_train_labels = np.eye(classes)[training_labels] 
    one_hot_test_labels = np.eye(classes)[test_labels]  

    training_images, validation_images = training_images[0:50000,:], training_images[50000:,:]
    training_labels, validation_labels = training_labels[0:50000], training_labels[50000:]
    one_hot_train_labels, one_hot_validation_labels = one_hot_train_labels[0:50000,:], one_hot_train_labels[50000:,:]
    
    nn = Network([784, 64, 10])
    nn.train(training_images, one_hot_train_labels, training_labels, test_images, one_hot_test_labels, test_labels, validation_images, validation_labels, one_hot_validation_labels)
