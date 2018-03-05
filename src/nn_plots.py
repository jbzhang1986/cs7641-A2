"""
Code to generate plots related to neural network weight optimization
experiments.

"""
from helpers import get_abspath
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def combine_datasets(datafiles):
    """Creates combined datasets for error and accuracy to compare the various
    optimization algorithms.

    Args:
        datafiles (dict(str)): Dictionary of filepaths to be used.

    """
    # create combined error datasets
    resdir = 'results/NN'
    BP = datafiles['BP']
    RHC =datafiles['RHC']
    SA = datafiles['SA']
    GA = datafiles['GA']

    # rename columns
    bpCols = {'MSE_train': 'bp_msetrain', 'MSE_test': 'bp_msetest', 'MSE_validation':'bp_msevalid', 'acc_train': 'bp_acctrain','acc_test':'bp_acctest', 'acc_validation':'bp_accvalid', 'seconds_elapsed': 'bp_time'}
    rhcCols = {'MSE_train': 'rhc_msetrain', 'MSE_test': 'rhc_msetest', 'MSE_validation':'rhc_msevalid', 'acc_train': 'rhc_acctrain','acc_test':'rhc_acctest', 'acc_validation':'rhc_accvalid', 'seconds_elapsed': 'rhc_time'}
    saCols = {'MSE_train': 'sa_msetrain', 'MSE_test': 'sa_msetest', 'MSE_validation':'sa_msevalid', 'acc_train': 'sa_acctrain','acc_test':'sa_acctest', 'acc_validation':'sa_accvalid', 'seconds_elapsed': 'sa_time'}
    gaCols = {'MSE_train': 'ga_msetrain', 'MSE_test': 'ga_msetest', 'MSE_validation':'ga_msevalid', 'acc_train': 'ga_acctrain','acc_test':'ga_acctest', 'acc_validation':'ga_accvalid', 'seconds_elapsed': 'ga_time'}

    BP.rename(index=str, columns=bpCols, inplace=True)
    RHC.rename(index=str, columns=rhcCols, inplace=True)
    SA.rename(index=str, columns=saCols, inplace=True)
    GA.rename(index=str, columns=gaCols, inplace=True)

    # create combined validation datasets ()


def combined_error(df, ef='Mean squared error'):
    """Generates plots for comparing error across the various optimization
    algorithms.

    Args:
        df (Pandas.DataFrame): Dataset.
        ef (str): Name of loss function.

    """
    print('hello')


def combined_acc(df):
    """Generates plots for comparing accuracy scores across the various
    optimization algorithms.

    Args:
        df (Pandas.DataFrame): Dataset.

    """
    print('hello')


def error_curve(df, oaName, title, ef='Mean squared error'):
    """Plots the error curve for a given optimization algorithm on the
    seismic-bumps dataset and saves it as a PNG file.

    Args:
        df (Pandas.DataFrame): Dataset.
        oaName (str): Name of optimization algorithm.
        title (str): Plot title.
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
    plt.plot(iterations, MSE_test, color='r', label='Test')
    plt.plot(iterations, MSE_valid, color='g', label='Validation')
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


def validation_curve(df, oaName, title):
    """Plots the validation curve for a given optimization algorithm on the
    seismic-bumps dataset and saves it as a PNG file.

    Args:
        df (Pandas.DataFrame): Dataset.
        oaName (str): Name of optimization algorithm.
        title (str): Plot title.

    """

    # get columns
    iterations = df['iteration']
    acc_train = df['acc_train']
    acc_test = df['acc_test']
    acc_valid = df['acc_validation']

    # create validation curve
    plt.figure(0)
    plt.plot(iterations, acc_train, color='b', label='Training')
    plt.plot(iterations, acc_test, color='r', label='Test')
    plt.plot(iterations, acc_valid, color='g', label='Validation')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.title('{} - Validation Curve'.format(title))
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    # save learning curve plot as PNG
    plotdir = 'plots/NN'
    plot_tgt = '{}/{}'.format(plotdir, oaName)
    plotpath = get_abspath('{}_VC.png'.format(oaName), plot_tgt)
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
    validation_curve(BP, oaName='BP', title='Backpropagation')
    validation_curve(RHC, oaName='RHC', title='Randomized Hill Climbing')
    validation_curve(SA, oaName='SA', title='Simulated Annealing')
    validation_curve(GA, oaName='GA', title='Genetic Algorithms')

    # generate combined plots
    datafiles = {'BP': BP, 'RHC': RHC, 'SA': SA, 'GA': GA}
    combine_datasets(datafiles)