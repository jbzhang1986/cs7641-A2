"""
Code to generate plots related to neural network weight optimization
experiments.

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from helpers import get_abspath

def combine_datasets(datafiles):
    """Creates combined datasets for error and accuracy to compare the various
    optimization algorithms.

    Args:
        datafiles (list(str)): List of files to be used.

    """
    # create combined error datasets



def combined_error(df, ef=)
def combined_acc(df, ef)


def error_curve(df, oaName, ef='Mean squared error'):
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
    MSE_valid =df['MSE_validation']

    # create error curve
    plt.figure(0)
    plt.plot(iterations, MSE_train, color='b', label='Training')
    plt.plot(iterations, MSE_test, color='r', label='Validation')
    plt.plot(iterations, MSE_valid, color='g', label='Test')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.xlabel('Iterations')
    plt.ylabel(ef)

    # save learning curve plot as PNG
    plotdir = 'plots/NN'
    plot_tgt = '{}/{}'.format(plotdir, oaName)
    plotpath = get_abspath('{}_error.png'.format(oaName), plot_tgt)
    plt.savefig(plotpath, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # generate individual algorithm plots

    # generate combined plots
