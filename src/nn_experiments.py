"""
Neural network weight optimization experiments. Algorithms used include
backpropagation (control variable), randomized hill climbing,

"""

import nn_base
from helpers import get_abspath


if __name__ == '__main__':
    # load dataset as instances
    filepath = 'data/experiments'
    filename = 'seismic_bumps.csv'
    input_file = get_abspath(filename, filepath)

    # set NN parameters
    input_layer = 21  # number of features
    hidden_layer = 5  # hidden layer nodes
    output_layer = 1  # output layer is always 1 for binomial classification
    training_iterations = 100

    respath = 'NN'
    # with open()
