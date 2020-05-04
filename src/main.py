import argparse
import time

import pandas as pd

from src.classifiers import Classifier
from src.data_manipulations import *
from src.data_visualisation import *
from src.helpers import load_model, print_error_message, print_runtime, save_df_to_pickle


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which agent to run.
    :return: None
    """
    parse_command_line_arguments()

    if config.verbose_mode:
        print("Verbose mode: ON\n")

    # Make final predictions on X_test.csv (produce 'Y_test.csv' for practical submission).
    if config.section == "test":
        X_test = load_testing_data()

        # Run this once to save the CSV files imported into DFs in PKL format for quicker loading times.
        # save_df_to_pickle(X_test, config.dataset, "X_test")

        X = input_preparation(X_test)
        make_final_test_predictions(X)

    # Explore and train the classifiers.
    else:
        X_train, y_train = load_training_data()

        # Run this once to save the CSV files imported into DFs in PKL format for quicker loading times.
        # save_df_to_pickle(X_train, config.dataset, "X_train")
        # save_df_to_pickle(y_train, config.dataset, "y_train")

        # Visualise data.
        if config.section == "data_vis":
            start_time = time.time()
            # Split training dataset's features in 3 distinct DFs.
            X_train_HoG, X_train_normal_dist, X_train_colour_hists = split_features(X_train)
            # Visualise data.
            data_overview(X_train)
            visualise_hog(X_train_HoG)
            visualise_rgb_hist(X_train_colour_hists)
            visualise_class_distribution(y_train)
            visualise_correlation(X_train, y_train)
            print_runtime(round(time.time() - start_time, 2))

        # Train classification models.
        elif config.section == "train":

            # Over sample binary dataset.
            if config.dataset == "binary":
                X_train, y_train = over_sample(X_train, y_train)

            # Run this once to save the CSV files imported into DFs in PKL format for quicker loading times.
            # save_df_to_pickle(X_train_resampled, config.dataset, "X_train_resampled")
            # save_df_to_pickle(y_train_resampled, config.dataset, "Y_train_resampled")
            if config.verbose_mode:
                visualise_class_distribution(X_train)

            X = input_preparation(X_train)
            y, ground_truth = output_preparation(y_train)
            if config.section == "train":
                train_classification_models(X, y, ground_truth)

        else:
            print_error_message()


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
                        help="The regression model to use for training. Must be either 'sgd', 'logistic', 'svc_lin',"
                             "'svc_poly', 'mlp' or 'dt'."
                        )
    parser.add_argument("-gs", "--gridsearch",
                        action="store_true",
                        default=False,
                        help="Include this flag to run the grid search algorithm to determine the optimal "
                             "hyperparameters for the classification model. Only works for multi layer perceptrons "
                             "(MLP - neural networks)."
                        )
    parser.add_argument("-rs", "--randomisedsearch",
                        action="store_true",
                        default=False,
                        help="Include this flag to run the randomised search algorithm to determine optimal "
                             "hyperparameters for the classification model. Only works for multi layer perceptrons "
                             "(MLP - neural networks)."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    args = parser.parse_args()
    config.section = args.section
    config.dataset = args.dataset
    config.model = args.model
    config.is_grid_search = args.gridsearch
    config.is_randomised_search = args.randomisedsearch
    config.verbose_mode = args.verbose


def train_classification_models(X, y, ground_truth) -> None:
    """

    :param X:
    :param y:
    :param ground_truth:
    :return:
    """
    # Start recording time.
    start_time = time.time()

    # Create classifier model instance.
    Classifier(config.model, X, y, ground_truth)

    # Print training runtime.
    print_runtime(round(time.time() - start_time, 2))


def make_final_test_predictions(X_test) -> None:
    """
    Make predictions on X_test and save them in 'Y_test.csv'.
    :param X_test: the samples to predict.
    :return: None
    """
    final_model = load_model(config.dataset, "mlp")
    predictions = final_model.predict(X_test)
    predictions_df = pd.DataFrame(predictions, index=None, columns=None)
    predictions_df.to_csv("../data/{}/Y_test.csv".format(config.dataset), index=False, columns=None, header=False)


if __name__ == "__main__":
    main()
