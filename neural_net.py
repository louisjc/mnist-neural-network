# encoding=utf8
# -*- coding: utf-8 -*-
"""
2 layers Neural network applied to handwriting recognition
from MNIST database.
"""

from __future__ import division
import time
import pickle
import gzip
from random import randint
from scipy import misc
from scipy import special
import numpy as np


# =====================
#    Initialisation
# =====================

# Initialisation - Import from MNIST database
START_TIME = time.time()
ft = gzip.open('data_training', 'rb')
TRAINING = pickle.load(ft)
ft.close()
ft = gzip.open('data_testing', 'rb')
TESTING = pickle.load(ft)
ft.close()

print('Import duration '+str(round((time.time() - START_TIME), 2))+'s')
print('----')

# =====================
#     Network class
# =====================


class Network:

    def __init__(self, num_hidden):
        self.input_size = 784
        self.output_size = 10
        self.num_hidden = num_hidden
        self.best = 0.
        self.same = 0

        # Standardize random weights
        # np.random.seed(0)
        hidden_layer = np.random.rand(self.num_hidden, self.input_size + 1) / self.num_hidden
        output_layer = np.random.rand(self.output_size, self.num_hidden + 1) / self.output_size
        self.layers = [hidden_layer, output_layer]
        self.iteration = 0.

        print('Initialization with random weight')
        print('-----')

    def train(self, batchsize, training):
        start_time = time.time()
        print('Network training with '+str(batchsize)+' examples')
        print('Until convergence (10 iterations without improvements)')
        print('-----')
        inputs = training[0][0:batchsize]
        targets = np.zeros((batchsize, 10))
        for i in range(batchsize):
            targets[i, training[1][i]] = 1

        # Performs iterations
        while self.same < 10:
            for input_vector, target_vector in zip(inputs, targets):
                self.backpropagate(input_vector, target_vector)
            # Messages and backups
            self.iteration += 1.
            accu = self.accu(TESTING)
            message = 'Iteration '+str(int(self.iteration)).zfill(2) + \
                ' (' + str(round(time.time()-start_time)).zfill(2)+'s) '
            message += 'Precision G:'+str(accu[1]).zfill(4)+'% Min:'+ \
                str(accu[0]).zfill(4)+ '% ('+str(int(accu[2]))+')'
            if accu[0] > self.best:
                self.same = 0
                self.best = accu[0]
                message += ' R'
                if accu[0] > 97:
                    self.sauv(file_name='ntMIN_'+str(accu))
                    message += 'S'
            else:
                self.same += 1
            print(message)
        # Final message
        print('10 Iterations without improvements.')
        print('Total duration: ' + str(round((time.time() - start_time), 2))+'s')

    def feed_forward(self, input_vector):
        """Takes a network (Matrix list) and returns the outputs of both
         layers by propagating the entry"""
        outputs = []
        for layer in self.layers:
            input_with_bias = np.append(input_vector, 1)   # Ajout constante
            output = np.inner(layer, input_with_bias)
            output = special.expit(output)
            outputs.append(output)
            # The output is the input of the next layer
            input_vector = output
        return outputs

    def backpropagate(self, input_vector, target):
        """Reduce error for one input vector:
        Calculating the partial derivatives for each coeff then subtracts"""
        c = 1./(self.iteration + 10)  # Learning coefficient
        hidden_outputs, outputs = self.feed_forward(input_vector)

        # Calculation of partial derivatives for the output layer and subtraction
        output_deltas = outputs * (1 - outputs) * (outputs - target)
        self.layers[-1] -= c*np.outer(output_deltas, np.append(hidden_outputs, 1))

        # Calculation of partial derivatives for the hidden layer and subtraction
        hidden_deltas = hidden_outputs * (1 - hidden_outputs) * \
            np.dot(np.delete(self.layers[-1], 200, 1).T, output_deltas)
        self.layers[0] -= c*np.outer(hidden_deltas, np.append(input_vector, 1))

    def predict(self, input_vector):
        return self.feed_forward(input_vector)[-1]

    def predict_one(self, input_vector):
        return np.argmax(self.feed_forward(input_vector)[-1])

    def sauv(self, file_name=''):
        if file_name == '':
            file_name = 'nt_'+str(self.accu(TESTING)[0])
        sauvfile = self.layers
        f = open(file_name, 'wb')
        pickle.dump(sauvfile, f)
        f.close()

    def load(self, file_name):
        f = open(file_name, 'rb')
        self.layers = pickle.load(f, encoding='latin1')
        f.close()

    def accu(self, testing):
        """The lowest precision digit and total"""
        res = np.zeros((10, 2))
        for k in range(len(testing[1])):
            if self.predict_one(testing[0][k]) == testing[1][k]:
                res[testing[1][k]] += 1
            else:
                res[testing[1][k]][1] += 1
        total = np.sum(res, axis=0)
        each = [res[k][0]/res[k][1] for k in range(len(res))]
        min_c = sorted(range(len(each)), key=lambda k: each[k])[0]
        return np.round([each[min_c]*100, total[0]/total[1]*100, min_c], 2)

nt1=Network(300)
nt1.train(600,TRAINING)

# =====================
#   Display fonctions
# =====================

# Rounding off the prints and scientific notation
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)


def find(c, network):
    x = randint(0, 999)
    while TESTING[1][x] != c:
        x = randint(0, 10000)
    aff(x, network)


def aff(x, network):
    print('Display character #'+str(x))
    print('Target = '+str(TESTING[1][x]))
    char = TESTING[0][x]
    l = ''
    for i in range(784):
        if i % 28 == 0:
            print(l)
            l = str(int(round(char[i])))
        else:
            l += str(int(round(char[i])))
    pred = network.predict(char)
    print('Prediction = ' + str(np.argmax(pred)))
    print(pred)


def err(network):
    x = randint(0, 10000)
    while network.predict_one(TESTING[0][x]) == TESTING[1][x]:
        x = randint(0, 10000)
    aff(x, network)


def test_nn(network):
    """Test Network"""
    ok, nb = 0, 10000
    for k in range(nb):
        if network.predict_one(TESTING[0][k]) == TESTING[1][k]:
            ok += 1
    return round((ok*100./nb), 1)
