import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspin.spin import Box1, make_spin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import src.config as config
from src.helpers import is_file_exists, save_plot


@make_spin(Box1, "Loading data into memory...")
def load_data(dataset):
    # If PKL format already exists for the data, load it for quicker loading times
    #   (generate it by uncommenting the call to save_df_to_pickle).
    if is_file_exists("../data/{}/X_train.pkl".format(dataset)):
        X_train = pd.read_pickle("../data/{}/X_train.pkl".format(dataset))
        y_train = pd.read_pickle("../data/{}/y_train.pkl".format(dataset))
        print("\nData loaded from 'X_train.pkl' and 'y_train.pkl'")
    # If PKL format not found, loads CSV file into memory (slower loadings times).
    else:
        X_train = pd.read_csv("../data/{}/X_train.csv".format(dataset), header=None)
        y_train = pd.read_csv("../data/{}/Y_train.csv".format(dataset), header=None)
        print("\nData loaded from 'X_train.csv' and 'y_train.csv'")
    return X_train, y_train


def split_features(X_train):
    # First 900 columns = HoG extracted from the image (10×10 px cells, 9 orientations, 2×2 blocks).
    X_train_HoG = X_train.iloc[:, :900]

    # Next 16 columns drawn from a normal distribution (µ = 0.5, σ = 2)
    X_train_normal_dist = X_train.iloc[:, 900:916]

    # Last 48 columns correspond to RGB colour histograms extracted from the same image with 16 bins per channel.
    X_train_colour_hists = X_train.iloc[:, 916:]

    return X_train_HoG, X_train_normal_dist, X_train_colour_hists


@make_spin(Box1, "Transforming features...")
def input_preparation(X_train, variance=0.99):
    # Standard scale inputs
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns.values)

    pca = PCA(n_components=487)  # n_components is determined by running the code below.
    X_train_reduced = pca.fit_transform(X_train_df)

    if config.verbose_mode:
        print("Dataset size before PCA: {}".format(X_train_df.shape))
        print("Dataset size after PCA: {}".format(X_train_reduced.shape))

    # Determine the number of dimensions to reduce the data to.
    if config.verbose_mode:
        # Perform PCA without reducing dimensionality
        pca = PCA()
        pca.fit(X_train_df)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= variance) + 1
        print("Dimensions to reduce to to preserve 99% explained variance: d={}\n".format(d))

        # Modified plot code, originally from Hands-On ML jupyter notebook, chapter 08 (In [35])
        # source: https://github.com/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb
        plt.figure(figsize=(6, 4))
        plt.plot(cumsum, linewidth=2, color="green")
        plt.axis([0, 964, 0, 1])
        plt.xlabel("Dimensions")
        plt.ylabel("Explained Variance")
        plt.plot([d, d], [0, variance], "k:", color="green", alpha=0.5)
        plt.plot([0, d], [variance, variance], "k:", color="green", alpha=0.5)
        plt.plot(d, variance, "ko")
        plt.grid(True)
        save_plot("explained_variance_plot_{}".format(config.dataset))
        plt.show()

    print("\nFeatures transformed using standard scaling and reduced using PCA")
    return X_train_reduced


def output_preparation(y_train):
    if config.dataset == "binary":
        # Convert class ID output to boolean integer format for logistic regression:
        #   1 if it's a "seal", 0 if it's a "background").
        y_train = (y_train == "seal").astype(np.int)

    # Get values in an array of shape (n,1) and then use ravel() to the convert that array shape to (n, ).
    y_train_unravelled = y_train.values.ravel()

    return y_train_unravelled, y_train
