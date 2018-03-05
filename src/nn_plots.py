"""
Code to generate plots related to neural network weight optimization
experiments.

"""
from helpers import get_abspath
import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def combine_datasets(datafiles):
    """Creates combined datasets for error and accuracy to compare the various
    optimization algorithms.

    Args:
        datafiles (list(str)): List of files to be used.

    """
    # create combined error datasets


def combined_error(df, ef='Mean squared error'):
    """Generates plots for comparing error across the various optimization
    algorithms.

    Args:
        df (Pandas.DataFrame): Dataset.
        ef (str): Name of loss function.

    """
    print('hello')


def combined_acc(df, ef='Mean squared error'):
    """Generates plots for comparing accuracy scores across the various
    optimization algorithms.

    Args:
        df (Pandas.DataFrame): Dataset.
        ef (str): Name of loss function.

    """
    print('hello')


def error_curve(df, oaName, title, ef='Mean squared error'):
    """Plots the error curve for a given optimization algorithm on the
    seismic-bumps dataset and saves it as a PNG file.

    Args:
        df (Pandas.DataFrame): Dataset.
        oaName (str): Name of optimization algorithm.
        ef (str): Name of loss function.

    """

    # get columns
    iterations = df['iteration']
    MSE_train = df['MSE_train']
    MSE_test = df['MSE_test']
    MSE_valid = df['MSE_validation']

    # create error curve
    plt.figure(0)
    plt.plot(iterations, MSE_train, color='b', label='Training')
    plt.plot(iterations, MSE_test, color='r', label='Validation')
    plt.plot(iterations, MSE_valid, color='g', label='Test')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.title('{} - {}'.format(title, ef))
    plt.xlabel('Iterations')
    plt.ylabel(ef)

    # save learning curve plot as PNG
    plotdir = 'plots/NN'
    plot_tgt = '{}/{}'.format(plotdir, oaName)
    plotpath = get_abspath('{}_error.png'.format(oaName), plot_tgt)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # get datasets for error and accuracy curves
    resdir = 'results/NN'
    BP = pd.read_csv(get_abspath('BP/results.csv', resdir))
    RHC = pd.read_csv(get_abspath('RHC/results.csv', resdir))
    SA = pd.read_csv(get_abspath('SA/results_10000000000.0_0.15.csv', resdir))
    GA = pd.read_csv(get_abspath('GA/results_100_10_10.csv', resdir))

    # generate individual algorithm error and accuracy curves
    error_curve(BP, oaName='BP', title='Backpropagation')
    error_curve(RHC, oaName='RHC', title='Randomized Hill Climbing')
    error_curve(SA, oaName='SA', title='Simulated Annealing')
    error_curve(GA, oaName='GA', title='Genetic Algorithms')

    # generate combined plots
