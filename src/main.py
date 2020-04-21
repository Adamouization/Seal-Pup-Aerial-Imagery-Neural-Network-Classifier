import argparse
import time

import numpy as np
from pyspin.spin import Box1, make_spin

from src.classifiers import Classifier
from src.data_vis import *
from src.helpers import print_error_message, print_runtime


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which agent to run.
    :return: None
    """
    parse_command_line_arguments()

    if config.verbose_mode:
        print("Verbose mode: ON\n")

    X_train, y_train = load_data(config.dataset)

    # Split training dataset's features:
    #   first 900 columns = HoG extracted from the image (10×10 px cells, 9 orientations, 2×2 blocks).
    #   next 16 columns drawn from a normal distribution (µ = 0.5, σ = 2)
    #   last 48 columns correspond to RGB colour histograms extracted from the same image with 16 bins per channel.
    X_train_HoG = X_train.iloc[:, :900]
    X_train_normal_dist = X_train.iloc[:, 900:916]
    X_train_colour_hists = X_train.iloc[:, 916:]

    if config.section == "data_vis":
        # Start recording time.
        start_time = time.time()

        visualise_hog(X_train_HoG)
        visualise_rgb_hist(X_train_colour_hists)
        visualise_class_distribution(y_train)
        # visualise_correlation(X_train, y_train)

        # Print training runtime.
        print_runtime(round(time.time() - start_time, 2))
    elif config.section == "train" or config.section == "test":
        X, y, ground_truth = input_preparation(X_train, y_train)
        if config.section == "train":
            train_classification_models(X, y, ground_truth)
        elif config.section == "test":
            pass
            # final_evaluation()
        else:
            print_error_message()
    else:
        print_error_message()

    pass


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save them in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--section",
                        required=True,
                        help="The section of the code to run. Must be either 'data_vis', 'train' or 'test'."
                        )
    parser.add_argument("-d", "--dataset",
                        required=True,
                        help="The dataset to use. Must be either 'binary' or 'multi'."
                        )
    parser.add_argument("-m", "--model",
                        default="logistic",
                        help="The regression model to use for training. Must be either 'sgd', 'logistic', 'svc_lin' or "
                             "'svc_poly'."
                        )
    parser.add_argument("-g", "--gridsearch",
                        action="store_true",
                        default=False,
                        help="Include this flag to run the grid search algorithm to determine the optimal "
                             "hyperparameters for the regression model. Only works for linear regression with either"
                             "Ridge or Lasso regularisation."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    args = parser.parse_args()
    config.section = args.section
    config.model = args.model
    config.is_grid_search = args.gridsearch
    config.verbose_mode = args.verbose


@make_spin(Box1, "Loading data into memory...")
def load_data(dataset):
    X_train = pd.read_csv("../data/{}/X_train.csv".format(dataset), header=None)
    y_train = pd.read_csv("../data/{}/Y_train.csv".format(dataset), header=None)
    return X_train, y_train


def input_preparation(X_train, y_train):
    # Convert class ID output to boolean integer format for logistic regression:
    #   1 if it's a "seal", 0 if it's a "background").
    y_train_seal = (y_train == "seal").astype(np.int)

    # Get values in an array of shape (n,1) and then use ravel() to the convert that array shape to (n, ).
    y_train_seal_unravelled = y_train_seal.values.ravel()

    return X_train, y_train_seal_unravelled, y_train_seal


def train_classification_models(X, y, ground_truth):
    # Start recording time.
    start_time = time.time()

    # Create classifier model instance.
    clf = Classifier(config.model, X, y, ground_truth)

    # Training pipeline.
    clf.fit_classifier()
    clf.k_fold_cross_validation()

    # Print training runtime.
    print_runtime(round(time.time() - start_time, 2))


def final_evaluation(test_set, final_model):
    pass


if __name__ == "__main__":
    main()
