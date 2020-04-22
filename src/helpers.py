import joblib
import os

import matplotlib.pyplot as plt


def print_error_message():
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


def save_plot(title):
    plt.savefig("../results/{}.png".format(title), bbox_inches='tight')


def save_model(model, dataset, model_type):
    """
    Function to save the model to a file.
    :param model:
    :param model_type:
    :return:
    """
    joblib.dump(model, "../trained_classifiers/{}_{}.pkl".format(dataset, model_type))


def load_model(dataset, model_type):
    """
    Function to load model.
    :param model_type:
    :return:
    """
    return joblib.load("../trained_classifiers/{}_{}.pkl".format(dataset, model_type))


def get_classifier_name(model):
    if model == "sgd":
        return "SGD Classifier"
    elif model == "logistic":
        return "Logistic Regression"
    elif model == "svc_lin":
        return "Linear SVM Classifier"
    elif model == "svc_poly":
        return "Polynomial (^2) SVM Classifier"


def is_trained_model_exists(dataset, model_type):
    path = "../trained_classifiers/{}_{}.pkl".format(dataset, model_type)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return True
    return False
