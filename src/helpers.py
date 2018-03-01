from __future__ import division
import os


def save_dataset(df, filename, sep=',', subdir='data'):

    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    df.to_csv(path_or_buf=tdir, sep=sep, header=True, index=False)


def get_abspath(filename, filepath):
    p = os.path.abspath(os.path.join(os.curdir, os.pardir))
    filepath = os.path.join(p, filepath, filename)

    return filepath
