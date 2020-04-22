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
from src.helpers import get_classifier_name, is_trained_model_exists, load_model, save_model, save_plot

kwargs = {"random_state": config.RANDOM_SEED}


class Classifier:

    def __init__(self, model, X, y, ground_truth):
        self.X = X
        self.y = y
        self.ground_truth = ground_truth
        self.folds = 3
        self.predictions = None

        # Load previously trained model.
        if is_trained_model_exists(config.dataset, config.model):
            print("Loaded previously trained classifier.")
            self.clf = load_model(config.dataset, config.model)
        # Create new sklearn model instance and fit it.
        else:
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
            self.fit_classifier()

        self.k_fold_cross_validation()
        self.evaluate_classifier()

    @make_spin(Box1, "Fitting {}...".format(get_classifier_name(config.model)))
    def fit_classifier(self):
        self.clf.fit(self.X, self.y)
        save_model(self.clf, config.dataset, config.model)

    @make_spin(Box1, "Making predictions...")
    def single_prediction(self):
        self.predictions = self.clf.predict(self.X)

    @make_spin(Box1, "Performing k-fold cross validation...")
    def k_fold_cross_validation(self):
        self.predictions = cross_val_predict(self.clf, self.X, self.y, cv=self.folds)

    def evaluate_classifier(self):
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

        precision = precision_score(self.ground_truth, self.predictions, average=avg_param)
        print("Precision: " + str(precision))
        recall = recall_score(self.ground_truth, self.predictions, average=avg_param)
        print("Recall: " + str(recall))
        f1 = f1_score(self.ground_truth, self.predictions, average=avg_param)
        print("F1: " + str(f1))
        scores_report = classification_report(self.ground_truth, self.predictions)
        print(scores_report)

        cm = confusion_matrix(self.ground_truth, self.predictions)
        _plot_pretty_confusion_matrix(cm, cm_labels, False)
        _plot_pretty_confusion_matrix(cm, cm_labels, True)


def _plot_pretty_confusion_matrix(cm, labels: list, is_normalised: bool) -> None:
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
    save_plot("{}_{}_class_distribution".format(config.dataset, config.model))
    plt.show()
