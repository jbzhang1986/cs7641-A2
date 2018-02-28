from scipy.io import arff
from helpers import get_abspath
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


def save_array(array, filename, sep=',', subdir='data'):
    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    np.savetxt(fname=tdir, X=array, delimiter=sep, fmt='%.20f')


def preprocess_seismic():
    """Cleans and generates seismic bumps dataset for experiments as a
    CSV file. Uses one-hot encoding for categorical features.

    """
    # get file path
    sdir = 'data/raw'
    tdir = 'data/experiments'
    seismic_file = get_abspath('seismic-bumps.arff', sdir)

    # read arff file and convert to record array
    rawdata = arff.loadarff(seismic_file)
    df = pd.DataFrame(rawdata[0])

    # apply one-hot encoding to categorical features using Pandas get_dummies
    cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
    cats = df[cat_cols]
    onehot_cols = pd.get_dummies(cats, prefix=cat_cols)

    # replace 0s with -1s to improve NN performance
    onehot_cols.replace(to_replace=[0], value=[-1], inplace=True)

    # drop original categorical columns and append one-hot encoded columns
    df.drop(columns=cat_cols, inplace=True)
    df = pd.concat((onehot_cols, df), axis=1)

    # drop columns that have only 1 unique value (features add no information)
    for col in df.columns:
        if len(np.unique(df[col])) == 1:
            df.drop(columns=col, inplace=True)

    # cast class column as integer
    df['class'] = df['class'].astype(int)

    # split out X data and scale (Gaussian zero mean and unit variance)
    X = df.drop(columns='class').as_matrix()
    y = df['class'].as_matrix()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    data = np.concatenate((X_scaled, y[:, np.newaxis]), axis=1)

    # save to CSV
    save_array(array=data, filename='seismic_bumps.csv', subdir=tdir)
    # np.savetxt(fname='seismic-bumps_X.csv', X=df_scaled, fmt='%.20f')


if __name__ == '__main__':
    # run preprocessing functions
    preprocess_seismic()
