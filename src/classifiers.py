import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspin.spin import Box1, make_spin
import seaborn as sns
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import src.config as config
from src.helpers import get_classifier_name, is_file_exists, load_model, save_model, save_plot

kwargs = {
    "random_state": config.RANDOM_SEED
}


class Classifier:
    """

    """

    def __init__(self, model: str, X, y, ground_truth):
        """

        :param model:
        :param X:
        :param y:
        :param ground_truth:
        """
        self.X = X
        self.y = y
        self.ground_truth = ground_truth
        self.folds = 3
        self.predictions = None

        # Load previously trained model.
        if is_file_exists("../trained_classifiers/{}_{}.pkl".format(config.dataset, config.model)):
            print("Loaded previously trained classifier.")
            self.clf = load_model(config.dataset, config.model)
        # Create new sklearn model instance and fit it.
        else:
            if model == "sgd":
                self.clf = SGDClassifier(**kwargs, penalty="l2", fit_intercept=True, max_iter=10000, tol=1e-3,
                                         n_jobs=-1)
            elif model == "logistic":
                self.clf = LogisticRegression(**kwargs, solver="liblinear", penalty="l2", fit_intercept=True,
                                              max_iter=100, tol=1e-4, n_jobs=-1)  # Uses OvR
            elif model == "svc_lin":
                self.clf = LinearSVC(**kwargs, multi_class="ovr", penalty="l2", fit_intercept=True, max_iter=1000,
                                     tol=1e-4)
            elif model == "svc_poly":
                self.clf = SVC(**kwargs, kernel='poly', degree=3, decision_function_shape="ovr", max_iter=1000,
                               tol=1e-3)
            elif model == "dt":
                self.clf = DecisionTreeClassifier(**kwargs, criterion="gini", splitter="best")
            elif model == "mlp":
                self.clf = MLPClassifier(**kwargs, hidden_layer_sizes=(100,), activation="relu", solver="adam",
                                         learning_rate="constant", learning_rate_init=0.6, momentum=0.9,
                                         max_iter=100, tol=1e-5, n_iter_no_change=15, verbose=config.verbose_mode)
            self.fit_classifier()

        self.k_fold_cross_validation()
        self.evaluate_classifier()

    @make_spin(Box1, "Fitting {}...".format(get_classifier_name(config.model)))
    def fit_classifier(self) -> None:
        """

        :return:
        """
        self.clf.fit(self.X, self.y)
        save_model(self.clf, config.dataset, config.model)

    @make_spin(Box1, "Making predictions...")
    def single_prediction(self) -> None:
        """

        :return:
        """
        self.predictions = self.clf.predict(self.X)

    @make_spin(Box1, "Performing k-fold cross validation...")
    def k_fold_cross_validation(self) -> None:
        """

        :return:
        """
        self.predictions = cross_val_predict(self.clf, self.X, self.y, cv=self.folds)

    def evaluate_classifier(self) -> None:
        """

        :return:
        """
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        print("Average accuracy over {} folds: {}%".format(self.folds, round(accuracy * 100, 2)))

        avg_param = str()
        cm_labels = list()
        if config.dataset == "binary":
            avg_param = "binary"
            cm_labels = ["Background", "Seal"]
        elif config.dataset == "multi":
            avg_param = "weighted"
            cm_labels = ["Background", "Dead Pups", "Juvenile", "Moulted Pup", "Whitecoat"]

        precision = round(precision_score(self.ground_truth, self.predictions, average=avg_param), 4)
        recall = round(recall_score(self.ground_truth, self.predictions, average=avg_param), 4)
        f1 = round(f1_score(self.ground_truth, self.predictions, average=avg_param), 4)
        scores_df = pd.DataFrame(np.array([[precision, recall, f1]]), columns=["precision", "recall", "f1"])
        print(scores_df)

        if config.verbose_mode:
            scores_report = classification_report(self.ground_truth, self.predictions)
            print(scores_report)

        cm = confusion_matrix(self.ground_truth, self.predictions)
        _plot_pretty_confusion_matrix(cm, cm_labels, False)
        _plot_pretty_confusion_matrix(cm, cm_labels, True)


def _plot_pretty_confusion_matrix(cm, labels: list, is_normalised: bool) -> None:
    """

    :param cm:
    :param labels:
    :param is_normalised:
    :return:
    """
    annot_format = "d"
    title = "{} data - {} Confusion matrix".format(config.dataset, get_classifier_name(config.model))

    if is_normalised:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot_format = ".2f"
        title += " (normalised)"

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig = plt.figure(figsize=(8, 5))
    heatmap = sns.heatmap(cm_df, cmap="YlGnBu", annot=True, fmt=annot_format)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0,
                                 ha='right',
                                 fontsize=config.fontsizes['ticks'])
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45,
                                 ha='right',
                                 fontsize=config.fontsizes['ticks'])
    plt.ylabel('True label', fontsize=config.fontsizes['axis'])
    plt.xlabel('Predicted label', fontsize=config.fontsizes['axis'])
    plt.title(title, fontsize=config.fontsizes['title'])

    if is_normalised:
        save_plot("{}_{}_class_distribution_normalised".format(config.dataset, config.model))
    else:
        save_plot("{}_{}_class_distribution".format(config.dataset, config.model))
    plt.show()
