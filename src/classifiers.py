import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspin.spin import Box1, make_spin
import seaborn as sns
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import src.config as config
from src.helpers import get_classifier_name, save_plot

kwargs = {"random_state": config.RANDOM_SEED}


class Classifier:

    def __init__(self, model, X, y, ground_truth):
        self.X = X
        self.y = y
        self.ground_truth = ground_truth

        if model == "sgd":
            self.clf = SGDClassifier(**kwargs, max_iter=1000, tol=1e-3)
        elif model == "logistic":
            self.clf = LogisticRegression(**kwargs, max_iter=1000, tol=1e-3)
        elif model == "svc_lin":
            self.clf = LinearSVC(**kwargs, max_iter=1000, tol=1e-3)
        elif model == "svc_poly":
            self.clf = SVC(**kwargs, kernel='poly', degree=2, max_iter=1000)
        elif model == "dt":
            self.clf = DecisionTreeClassifier(**kwargs, max_depth=5)
        elif model == "mlp":
            self.clf = MLPClassifier(**kwargs, hidden_layer_sizes=(15,), learning_rate_init=1, momentum=0.1,
                                     verbose=config.verbose_mode)

    @make_spin(Box1, "Fitting {}...".format(get_classifier_name(config.model)))
    def fit_classifier(self):
        self.clf.fit(self.X, self.y)

    @make_spin(Box1, "Performing k-fold cross validation...")
    def k_fold_cross_validation(self, folds=3):
        clf_accuracy_predictions = cross_val_predict(self.clf, self.X, self.y, cv=folds)

        cm = confusion_matrix(self.ground_truth, clf_accuracy_predictions)
        _plot_pretty_confusion_matrix(cm, ["Background", "Seal"], False)
        _plot_pretty_confusion_matrix(cm, ["Background", "Seal"], True)

        accuracy = accuracy_score(self.ground_truth, clf_accuracy_predictions)
        print("Average accuracy over {} folds: {}%".format(folds, round(accuracy * 100, 2)))


def _plot_pretty_confusion_matrix(cm, labels: list, is_normalised: bool) -> None:
    annot_format = "d"
    title = "{} Confusion matrix".format(get_classifier_name(config.model))
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
    save_plot("{}_binary_class_distribution".format(config.model))
    plt.show()
