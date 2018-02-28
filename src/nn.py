"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to find optimal weights to a neural network that is classifying the seismic-bumps dataset.
"""
import os
import csv
import time

from helpers import get_abspath

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm


def initialize_instances(input_file):
    """Read the abalone.txt CSV data into a list of instances."""
    instances = []

    # Read in the abalone.txt CSV file
    with open(input_file, "r") as seismic:
        reader = csv.reader(seismic)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) < 15 else 1))
            instances.append(instance)

    return instances


def train(oa, network, oaName, instances, measure, iterations):
    """Train a given network on a set of instances.
    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(iterations):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        print "%0.03f" % error


# def main():
#     """Run algorithms on the abalone dataset."""
#     instances = initialize_instances()
#     factory = BackPropagationNetworkFactory()
#     measure = SumOfSquaresError()
#     data_set = DataSet(instances)

#     networks = []  # BackPropagationNetwork
#     nnop = []  # NeuralNetworkOptimizationProblem
#     oa = []  # OptimizationAlgorithm
#     oa_names = ["RHC", "SA", "GA"]
#     results = ""

#     for name in oa_names:
#         classification_network = factory.createClassificationNetwork(
#             [INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
#         networks.append(classification_network)
#         nnop.append(NeuralNetworkOptimizationProblem(
#             data_set, classification_network, measure))

#     oa.append(RandomizedHillClimbing(nnop[0]))
#     oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
#     oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

#     for i, name in enumerate(oa_names):
#         start = time.time()
#         correct = 0
#         incorrect = 0

#         train(oa[i], networks[i], oa_names[i], instances, measure)
#         end = time.time()
#         training_time = end - start

#         optimal_instance = oa[i].getOptimal()
#         networks[i].setWeights(optimal_instance.getData())

#         start = time.time()
#         for instance in instances:
#             networks[i].setInputValues(instance.getData())
#             networks[i].run()

#             predicted = instance.getLabel().getContinuous()
#             actual = networks[i].getOutputValues().get(0)

#             if abs(predicted - actual) < 0.5:
#                 correct += 1
#             else:
#                 incorrect += 1

#         end = time.time()
#         testing_time = end - start

#         results += "\nResults for %s: \nCorrectly classified %d instances." % (
#             name, correct)
#         results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (
#             incorrect, float(correct) / (correct + incorrect) * 100.0)
#         results += "\nTraining time: %0.03f seconds" % (training_time,)
#         results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

#     print results


if __name__ == "__main__":
    filepath = 'data/experiments'
    filename = 'seismic_bumps.csv'
    input_file = get_abspath(filename, filepath)

    INPUT_LAYER = 22  # number of features
    HIDDEN_LAYER = 15  # hidden layer nodes
    OUTPUT_LAYER = 1  # output layer is always 1 for binomial classification
    TRAINING_ITERATIONS = 1000

    instances=initialize_instances(input_file)
    print instances[0]
    # main()
