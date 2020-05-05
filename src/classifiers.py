import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspin.spin import Box1, make_spin
from scipy.stats import randint as sp_randint, uniform as sp_uniform
import seaborn as sns
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import src.config as config
from src.helpers import get_classifier_name, save_model, save_plot

kwargs = {
    "random_state": config.RANDOM_SEED
}


class Classifier:
    """
    Classifier class containing functions fit, evaluate and perform grid search.
    """

    def __init__(self, model: str, X, y, ground_truth):
        """
        Initialise parameters.
        :param model: string specifying which sklearn model to use.
        :param X: processed data.
        :param y: processed labels.
        :param ground_truth: unprocessed labels.
        """
        self.X = X
        self.y = y
        self.ground_truth = ground_truth
        self.folds = 3
        self.predictions = None

        # Perform hyperparameter tuning (grid search) on the chosen model.
        if config.is_grid_search or config.is_randomised_search:
            self.clf = MLPClassifier(**kwargs, solver="adam", learning_rate="constant", activation="relu",
                                     verbose=config.verbose_mode)
            self.hyperparameter_tuning()

        # Fit, perform k-fold cross validation and evaluate a model.
        else:
            # Instantiate sklearn model.
            if model == "mlp":  # Selected model
                if config.dataset == "binary":
                    # MLP hyperparameters determined from optimal Randomised Search followed by Grid Search.
                    self.clf = MLPClassifier(**kwargs, activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                                             beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                             hidden_layer_sizes=(114, 114), learning_rate='constant',
                                             learning_rate_init=0.001, max_fun=15000, max_iter=200,
                                             momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                                             power_t=0.5, shuffle=True, solver='adam', tol=0.0001,
                                             validation_fraction=0.1, verbose=config.verbose_mode, warm_start=False)
                elif config.dataset == "multi":
                    # MLP hyperparameters determined from optimal Randomised Search followed by Grid Search.
                    self.clf = MLPClassifier(**kwargs, activation='relu', alpha=0.9, batch_size='auto', beta_1=0.9,
                                             beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                             hidden_layer_sizes=(100,), learning_rate='constant',
                                             learning_rate_init=0.001, max_fun=15000, max_iter=1000,
                                             momentum=0.1, n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                                             shuffle=True, solver='adam', tol=1e-05, validation_fraction=0.1,
                                             verbose=config.verbose_mode, warm_start=False)
            elif model == "sgd":
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

            # Fit and evaluate selected model.
            self.fit_classifier()
            self.k_fold_cross_validation()
            self.evaluate_classifier()

    @make_spin(Box1, "Fitting {}...".format(get_classifier_name(config.model)))
    def fit_classifier(self) -> None:
        """
        Fit the classifier and save the model using Pickle.
        :return:
        """
        self.clf.fit(self.X, self.y)
        save_model(self.clf, config.dataset, config.model)

    @make_spin(Box1, "Making predictions...")
    def single_prediction(self) -> None:
        """
        Make a prediction on the data.
        :return: None.
        """
        self.predictions = self.clf.predict(self.X)

    @make_spin(Box1, "Performing k-fold cross validation...")
    def k_fold_cross_validation(self) -> None:
        """
        Apply k-fold cross validation.
        :return: None.
        """
        self.predictions = cross_val_predict(self.clf, self.X, self.y, cv=self.folds)

    def evaluate_classifier(self) -> None:
        """
        Measure accuracy, precision, recall, f1-score and confusion matrix.
        :return: None.
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

    @make_spin(Box1, "Performing hyperparameter tuning...")
    def hyperparameter_tuning(self) -> None:
        """
        Performs a hyperparameter tuning search (either grid search or randomised search) on the defined parameters and
        saves the results in a CSV file for further analysis.
        Note: only designed to work with MLP (determined based on initial evaluations).
        :return: None.
        """
        # Determine scoring metric to use based on dataset.
        scoring = str()
        if config.dataset == "binary":
            scoring = "f1"
        elif config.dataset == "multi":
            scoring = "f1_weighted"

        parameters = dict()
        search_alg_str = str()
        # Initialise Grid Search.
        if config.is_grid_search:
            print("Hyperparameter tuning technique chosen: GRID SEARCH")
            if config.dataset == "binary":
                parameters = {
                    "hidden_layer_sizes": [(98,), (98, 98), (114,), (114, 114)],
                    "learning_rate_init": [0.001, 0.03, 0.04, 0.1],
                    "alpha": [0.0001, 0.26, 0.96]
                }
                print(parameters)
            elif config.dataset == "multi":
                parameters = {
                    "hidden_layer_sizes": [(68,), (68, 68), (100,), (100, 100)],
                    "learning_rate_init": [0.001, 0.01, 0.1],
                    "momentum": [0.1, 0.9],
                    "alpha": [0.0001, 0.1, 0.9]
                }
            searchCV = GridSearchCV(
                param_grid=parameters,
                estimator=self.clf,
                cv=self.folds,
                scoring=scoring
            )
            search_alg_str = "gs"
        # Initialise Randomised Search.
        elif config.is_randomised_search:
            print("Hyperparameter tuning technique chosen: RANDOMISED SEARCH")
            parameters = {
                'hidden_layer_sizes': (sp_randint(1, 150)),
                'learning_rate_init': sp_uniform(0.001, 1),
                'momentum': sp_uniform(0.1, 0.9),
                'alpha': sp_uniform(0.0001, 1)
            }
            searchCV = RandomizedSearchCV(
                param_distributions=parameters,
                estimator=self.clf,
                n_iter=100,
                cv=self.folds,
                scoring=scoring
            )
            search_alg_str = "rs"

        # Run the search and save results in a CSV file.
        gs_results = searchCV.fit(self.X, self.y)
        gs_results_df = pd.DataFrame(gs_results.cv_results_)
        gs_results_df.to_csv("../results/grid_search/{}_{}_{}.csv".format(config.dataset, config.model, search_alg_str))

        # Print the best model found by hyperparameter tuning algorithm for the MLP and save the model in a Pickle file.
        final_model = gs_results.best_estimator_
        print("\nBest model hyperparameters found by randomised search algorithm:")
        print(final_model)
        print("Score: {}".format(gs_results.best_score_))
        save_model(final_model, config.dataset,
                   "{}_{}_{}_best_estimator".format(config.dataset, config.model, search_alg_str))


def _plot_pretty_confusion_matrix(cm, labels: list, is_normalised: bool) -> None:
    """
    Plot the confusion matrix using seaborn to prettify the matrix.
    :param cm: the confusion matrix.
    :param labels: the labels to print.
    :param is_normalised: boolean to print a normalised or unnormalised matrix.
    :return: None.
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
