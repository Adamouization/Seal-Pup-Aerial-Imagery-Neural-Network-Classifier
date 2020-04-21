import joblib

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
    print("\n--- Runtime: {} seconds ---".format(runtime))


def save_plot(title):
    plt.savefig("../results/{}.png".format(title))


def save_model(model, model_type):
    """
    Function to save the model to a file.
    :param model:
    :param model_type:
    :return:
    """
    joblib.dump(model, "trained_classifiers/{}.pkl".format(model_type))


def load_model(model_type):
    """
    Function to load model.
    :param model_type:
    :return:
    """
    return joblib.load("trained_classifiers/{}.pkl".format(model_type))


def print_classifier_name(model):
    classifier_name = str()
    if model == "sgd":
        classifier_name = "Stochastic Gradient Descent"
    elif model == "logistic":
        classifier_name = "Logistic Regression"
    elif model == "svc_lin":
        classifier_name = "Linear Support Vector Machine Classifier"
    elif model == "svc_poly":
        classifier_name = "Polynomial (2nd degree) Support Vector Machine Classifier"
    else:
        print_error_message()
    print("Classifier: {}\n".format(classifier_name))


def get_classifier_name(model):
    if model == "sgd":
        return "Stochastic Gradient Descent Classifier"
    elif model == "logistic":
        return "Logistic Regression"
    elif model == "svc_lin":
        return "Linear Support Vector Machine Classifier"
    elif model == "svc_poly":
        return "Polynomial (2nd degree) Support Vector Machine Classifier"
