from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspin.spin import Box1, make_spin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import src.config as config
from src.helpers import is_file_exists, save_df_to_pickle, save_plot


@make_spin(Box1, "Loading training data into memory...")
def load_training_data():
    """
    Load the training data (input and labels) either from a serialised file or the initial raw CSV file.
    :return: the DataFrames holding the X_train and y_train datasets.
    """
    # If PKL format already exists for the data, load it for quicker loading times
    #   (generate it by uncommenting the call to save_df_to_pickle).
    if is_file_exists("../data/{}/X_train.pkl".format(config.dataset)):
        X_train = pd.read_pickle("../data/{}/X_train.pkl".format(config.dataset))
        y_train = pd.read_pickle("../data/{}/y_train.pkl".format(config.dataset))
        print("\nData loaded from 'X_train.pkl' and 'y_train.pkl'")

    # If PKL format not found, loads CSV file into memory (slower loadings times).
    else:
        X_train = pd.read_csv("../data/{}/X_train.csv".format(config.dataset), header=None)
        y_train = pd.read_csv("../data/{}/Y_train.csv".format(config.dataset), header=None)
        print("\nData loaded from 'X_train.csv' and 'y_train.csv'")
    return X_train, y_train


@make_spin(Box1, "Loading testing data into memory...")
def load_testing_data():
    """

    :param dataset:
    :return:
    """
    # If PKL format already exists for the data, load it for quicker loading times
    #   (generate it by uncommenting the call to save_df_to_pickle).
    if is_file_exists("../data/{}/X_test.pkl".format(config.dataset)):
        X_test = pd.read_pickle("../data/{}/X_test.pkl".format(config.dataset))
        print("\nData loaded from 'X_test.pkl'")

    # If PKL format not found, loads CSV file into memory (slower loadings times).
    else:
        X_test = pd.read_csv("../data/{}/X_test.csv".format(config.dataset), header=None)
        print("\nData loaded from 'X_test.csv'")
    return X_test


def split_features(X_train):
    """

    :param X_train:
    :return:
    """
    # First 900 columns = HoG extracted from the image (10×10 px cells, 9 orientations, 2×2 blocks).
    X_train_HoG = X_train.iloc[:, :900]

    # Next 16 columns drawn from a normal distribution (µ = 0.5, σ = 2)
    X_train_normal_dist = X_train.iloc[:, 900:916]

    # Last 48 columns correspond to RGB colour histograms extracted from the same image with 16 bins per channel.
    X_train_colour_hists = X_train.iloc[:, 916:]

    return X_train_HoG, X_train_normal_dist, X_train_colour_hists


@make_spin(Box1, "Oversampling the data using SMOTE algorithm...")
def over_sample(X_train, y_train):
    """
    Oversamples the dataset using the SMOTE algorithm.
    :param X_train: features DF.
    :param y_train: labels DF.
    :return: the oversampled dataset.
    """
    sm = SMOTE(random_state=config.RANDOM_SEED)
    X_train_resampled, y_train_resampled = sm.fit_sample(X_train, y_train)
    return X_train_resampled, y_train_resampled


@make_spin(Box1, "Transforming features...")
def input_preparation(X_train, variance: float = 0.99):
    """
    Data pre-processing for classification task:
        1) Drop normal distribution features (keep HoG and RGB histograms)
        2) Standardise data
        3) Apply PCA to reduce to 475 dimensions.
    :param X_train: the un-processed features.
    :param variance: the explained variance to conserve in the data after PCA.
    :return: the processed features, ready for the classification.
    """
    # Drop the normal distribution features (columns 900-916), only keep HoG and RGB histograms (0-900;916-964).
    X_train_trimmed = pd.concat([X_train.iloc[:, :900], X_train.iloc[:, 916:]], axis=1)

    # Standard scale inputs.
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train_trimmed)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train_trimmed.columns.values)

    # Apply PCA reduction.
    pca = PCA(n_components=475)  # n_components is determined by running the code below (487 with all features).
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
        plt.axis([0, X_train_trimmed.shape[1], 0, 1])  # X_train_trimmed.shape[1] is the number of dimensions (features)
        plt.xlabel("Dimensions")
        plt.ylabel("Explained Variance")
        plt.plot([d, d], [0, variance], "k:", color="green", alpha=0.5)
        plt.plot([0, d], [variance, variance], "k:", color="green", alpha=0.5)
        plt.plot(d, variance, "ko")
        plt.grid(True)
        save_plot("explained_variance_plot_{}".format(config.dataset))
        plt.show()

    # Run this once to save the CSV files imported into DFs in PKL format for quicker loading times.
    # save_df_to_pickle(X_train_reduced, config.dataset, "X_train_processed")

    print("\nFeatures transformed using standard scaling and reduced using PCA")
    return X_train_reduced


def output_preparation(y_train):
    """
    Prepare the labels for the classifiers.
    :param y_train: the DF holding the un-processed labels.
    :return: the unravelled and un-processed DFs holding the labels.
    """
    if config.dataset == "binary":
        # Convert class ID output to boolean integer format for logistic regression:
        #   1 if it's a "seal", 0 if it's a "background").
        y_train = (y_train == "seal").astype(np.int)

    # Get values in an array of shape (n,1) and then use ravel() to the convert that array shape to (n, ).
    y_train_unravelled = y_train.values.ravel()

    return y_train_unravelled, y_train


def revert_binary_predictions(predictions):
    return predictions
