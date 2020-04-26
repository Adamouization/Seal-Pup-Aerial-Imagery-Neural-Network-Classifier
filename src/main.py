import argparse
import time

from src.classifiers import Classifier
from src.data_manipulations import *
from src.data_visualisation import *
from src.helpers import print_error_message, print_runtime, save_df_to_pickle


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which agent to run.
    :return: None
    """
    parse_command_line_arguments()

    if config.verbose_mode:
        print("Verbose mode: ON\n")

    X_train, y_train = load_data(config.dataset)

    # Run this once to save the CSV files imported into DFs in PKL format for quicker loading times.
    # save_df_to_pickle(X_train, y_train, config.dataset)

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

    # Train or test classification models.
    elif config.section == "train" or config.section == "test":
        X = input_preparation(X_train)
        y, ground_truth = output_preparation(y_train)
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
    config.dataset = args.dataset
    config.model = args.model
    config.is_grid_search = args.gridsearch
    config.verbose_mode = args.verbose


def train_classification_models(X, y, ground_truth):
    # Start recording time.
    start_time = time.time()

    # Create classifier model instance.
    Classifier(config.model, X, y, ground_truth)

    # Print training runtime.
    print_runtime(round(time.time() - start_time, 2))


def final_evaluation(test_set, final_model):
    pass


if __name__ == "__main__":
    main()
