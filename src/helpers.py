import joblib
import os

import matplotlib.pyplot as plt
import pandas as pd


def print_error_message() -> None:
    """
    Print error message and exit code when a CLI-related error occurs.
    :return:
    """
    print("Wrong command line arguments passed, please use 'python main.py --help' for instructions on which arguments"
          "to pass to the program.")
    exit(1)


def print_runtime(runtime: float) -> None:
    """
    Print runtime in seconds.
    :param runtime: the runtime
    :return: None
    """
    print("\n--- Training Runtime: {} seconds ---".format(runtime))


def save_plot(title: str) -> None:
    """

    :param title:
    :return:
    """
    plt.savefig("../results/{}.png".format(title), bbox_inches='tight')


def save_model(model, dataset, model_type: str) -> None:
    """
    Function to save the model to a file.
    :param model:
    :param model_type:
    :return:
    """
    joblib.dump(model, "../trained_classifiers/{}_{}.pkl".format(dataset, model_type))


def save_transformation_pipeline(transform, dataset, transform_type: str) -> None:
    """
    Function to save the model to a file.
    :param transform:
    :param dataset:
    :param transform_type:
    :return:
    """
    joblib.dump(transform, "../transform_pipeline/{}_{}.pkl".format(dataset, transform_type))


def load_model(dataset, model_type: str):
    """
    Function to load model.
    :param dataset:
    :param model_type:
    :return:
    """
    return joblib.load("../trained_classifiers/{}_{}.pkl".format(dataset, model_type))


def load_transformation_pipeline(dataset, transform_type: str):
    """
    Function to load model.
    :param dataset:
    :param model_type:
    :return:
    """
    return joblib.load("../transform_pipeline/{}_{}.pkl".format(dataset, transform_type))


def get_classifier_name(model: str):
    if model == "sgd":
        return "SGD Classifier"
    elif model == "logistic":
        return "Logistic Regression"
    elif model == "svc_lin":
        return "Linear SVM Classifier"
    elif model == "svc_poly":
        return "Polynomial (^2) SVM Classifier"
    elif model == "dt":
        return "Decision Tree"
    elif model == "mlp":
        return "Neural Network"


def is_file_exists(filepath: str) -> bool:
    """
    Checks that the file exists and that it is not empty (more than 0 bytes).
    :param filepath:
    :return:
    """
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        return True
    return False


def save_df_to_pickle(df, dataset, name) -> None:
    """
    Save a DataFrame to a serialised Pickle format.
    :param df: the DataFrame to save.
    :param dataset: string representation of the dataset used for file organisation.
    :param name: string representation of the model name.
    :return: None.
    """
    pd.to_pickle(df, "../data/{}/{}.pkl".format(dataset, name))
    print("Saved DataFrame in PKL format.")
