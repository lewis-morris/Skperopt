import copy
import datetime
from random import shuffle
import random as rnd
import numpy
import pandas
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from hyperopt.base import STATUS_SUSPENDED
from hyperopt.base import STATUS_OK
from hyperopt.base import Trials
from hyperopt.fmin import fmin
from hyperopt import tpe
from hyperopt import hp


def scorer_is_better(test_type, new_score, old_score):
    if type(test_type) == list:
        test_type = test_type[0]

    if test_type in ["f1", "auc", "accuracy"]:
        if old_score <= new_score:
            return True
        else:
            return False
    elif test_type == "rmse":
        if old_score >= new_score:
            return True
        else:
            return False


def cross_validation(est, X, y, cv=5, scorer="f1", average_score=True, random=False, split_type="Stratified"):
    """
    Can give input as "Stratified" or "Kfold" for split type:

    :param est:
    :param X:
    :param y:
    :param cv:
    :param scorer:
    :param average_score:
    :return:
    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    """
    if hasattr(est, "best_score"):
        try:
            est.pickle_load_score()
        except:
            est.best_score = 0

    if split_type not in ["Stratified", "Kfold"]:
        print("Error setting split type, defaulting to Kfold")
        split_type = "Kfold"

    score_list = []

    if type(cv) == tuple:
        loops = cv[1]
    else:
        loops = 1

    for loo in range(0, loops):
        for X_train, X_test, y_train, y_test in get_splitter(split_type)(X, y, folds=cv, random=random):
            # reshape
            y_train = numpy.array(y_train).reshape(-1, 1).squeeze()
            y_test = numpy.array(y_test).reshape(-1, 1).squeeze().ravel()

            # fit -- try both methods

            est.fit(X_train, y_train)

            # predict
            pred = est.predict(X_test)

            # get score
            current_score = get_score(y_test, pred, scorer=scorer)

            # append
            score_list.append(current_score)

    if average_score:
        score = numpy.mean(score_list)
        # set best score on estimator if applicable
        return score
    else:
        return score_list


def StratifiedKFolds(X, y, folds=5, random=False):
    # join X & y

    X = change_to_df(X)
    y = change_to_df(y)

    # check if folds input is a tuple
    if type(folds) == tuple:
        folds = folds[0]

    df = pandas.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    # df = df.loc[random.shuffle(df.index.tolist())]
    df_dic = {}

    # split the data into classes so stratified folds can be made
    for x in df.iloc[:, -1].value_counts().index:
        index = df[df.iloc[:, -1] == x].index.values.tolist()
        rnd.seed(10)
        if random:
            rnd.shuffle(index)
        df_dic[x] = index

    # create start position dict
    start_dic = {k: 0 for k, v in df_dic.items()}

    # create df dict for concatenation

    # loop folds
    for x in range(1, folds + 1):
        dic_df = {}
        for k, v in df_dic.items():

            # get len per holdout fold
            each = int(len(v) / folds)

            # calculate fold index
            if x == folds:
                fold = v[start_dic[k]:]
            else:
                fold = v[start_dic[k]:each * x]

            # log fold for each class
            dic_df[k] = df.loc[fold]

            # increment start position for next loop
            start_dic[k] += each

        # concat df's and yield the folds
        test = pandas.concat(dic_df.values())
        train = df[df.index.isin(test.index) == False]

        if random:
            ind, ind1 = test.index.to_list(), train.index.to_list()
            shuffle(ind)
            shuffle(ind1)
            test = test.loc[ind]
            train = train.loc[ind1]

        yield train.iloc[:, 0:-1], test.iloc[:, 0:-1], train.iloc[:, -1::], test.iloc[:, -1::]


def get_score(y_true, y_pred, scorer):
    """
    Get score from predicitons and true Y data
    :param y_true:
    :param y_pred:
    :param scorer:
    :return:
    """
    score_list = []

    if type(scorer) == str:
        scorer = [scorer]

    for score_type in scorer:
        if score_type == "f1":
            score_list.append(f1_score(y_true, y_pred, average='macro'))
        elif score_type == "rmse":
            score_list.append(numpy.math.sqrt(((y_pred - y_true) ** 2).mean()))
        elif score_type == "auc":
            score_list.append(roc_auc_score(y_true, y_pred, average="macro"))
        elif score_type == "accuracy":
            score_list.append(accuracy_score(y_true, y_pred))
        else:
            return None
    return numpy.mean(score_list)


def get_splitter(split_type):
    """
    "Stratified" or "Kfold"

    :param split_type:
    :return:
    """
    if split_type == "Stratified":
        return StratifiedKFolds
    elif split_type == "Kfold":
        return KFoldSplits


def change_to_df(df):
    """Makes sure that the input data is dataframe format if not it is converted
    also makes sure that the column names are in string format"""
    if not hasattr(df, "iloc") or hasattr(df, "unique"):
        df = pandas.DataFrame(df)
    df.columns = [str(x) for x in df.columns.tolist()]
    return df


def KFoldSplits(X, y, folds=3, random=False):
    """
    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    KFold Split Generator

    Input expceted as array or dataframe

    Use as follows:

        for X_train,X_test,y_train,y_test in KFoldSPlits(X,y):
            do xxxx

    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    """

    if hasattr(X, "loc"):
        y = change_to_df(y)
        df = pandas.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        index = df.index.values.tolist()
        rnd.seed(10)
        if random:
            rnd.shuffle(index)
        df = df.loc[index]
        X = df.iloc[:, 0:-1]
        y = df.iloc[:, -1::]
    else:
        print("Input not a Pandas DataFrame or Pandas Series")
        return

    starting_rows = 0
    original_rows_per_fold = int(int(len(df)) / int(folds))
    rows_per_fold = original_rows_per_fold

    for x in range(folds):
        if x != 0 and x != folds - 1:
            starting_rows = rows_per_fold
            rows_per_fold = rows_per_fold + original_rows_per_fold
        elif x == folds - 1:
            starting_rows = rows_per_fold
            rows_per_fold = (rows_per_fold + ((len(df) + 1) - rows_per_fold)) - 1

        current_rows = [int(x) for x in numpy.arange(starting_rows, rows_per_fold)]

        X_test = X.iloc[current_rows]
        y_test = y.iloc[current_rows]
        X_train = X.loc[~X.index.isin(X_test.index)]
        y_train = y.loc[~y.index.isin(y_test.index)]

        yield X_train, X_test, y_train, y_test


def create_hyper(paramgrid):
    new_grid = {}
    for k, v in paramgrid.items():
        new_grid[k] = hp.choice(k, v)
    return new_grid


class HyperSearch:
    """
    A parameter searing algorithm wrapper that uses TPE to search for the best parameters,
    implements early stopping. Accepts sklearn style estimators as the input EST

    :param est:            class: sklearn style estimator
    :param X:              X data
    :param y:              y data
    :param iters:          number of iterations of searches to try - default 500 = no time contraint
    :param time_to_search: max time in seconds to try searching  - default None = no time contraint
    :param cv:             the number of folds to base the score on
    :param scorer:         the scoring method used
    :param verbose:        show information? 0 = no, > 0 increases verboseness
    :param params:         parameter grid

    Run with:
        HyperSearch.search()

    Once complete gather the best parameter grid with:
        HyperSearch.best_params

    View run stats with:
        HyperSearch.stats

    """

    def __init__(self, est, X, y, params=None, iters=500, time_to_search=None, cv=5, scorer="f1", verbose=1, random = False, foldtype = "Kfold"):

        # check and get skiperopt style parameters
        if params is None:
            params = {}
        if params == {}:
            raise ValueError("No parameters supplied")
        self.params = params
        self.best_params = None
        self.__space = create_hyper(params)
        self.initparams = est.get_params()
        # set hyper settings
        self.__algo = tpe.suggest
        self.__trial = Trials()

        # set run settings
        self.verbose = verbose
        self.iters = iters
        self.cv = cv
        self.scorer = scorer
        self.__run = 0

        self.est = est

        self.stats = {}
        self.__init_score = cross_validation(self.est, X, y, cv=self.cv, scorer=self.scorer)
        self.best_score = self.__init_score

        self.__X = X
        self.__y = y

        self.start_now = None

        self.__runok = True
        self.time_to_search = time_to_search

        self.random = random
        self.foldtype = foldtype

        if self.verbose > 0:
            print(f"Initial Score is {self.best_score}")

    def __objective(self, params):
        """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

        if self.__runok:
            est = copy.deepcopy(self.est)
            try:
                est.set_params(**params)
                best_score = cross_validation(est, self.__X, self.__y,
                                              cv=self.cv, scorer=self.scorer,
                                              split_type=self.foldtype,
                                              random=self.random)

                # Loss must be minimized
                if self.scorer in ["f1", "auc", "accuracy"]:
                    loss = 1 - best_score
                else:
                    loss = best_score

                # check if the latest score is best?
                if scorer_is_better(self.scorer, best_score, self.best_score):
                    self.best_score = copy.copy(best_score)
                    self.best_params = params
                    if self.verbose > 0 and best_score != self.best_score:
                        print(f"Found {best_score}")

                # log in the stats dict
                self.stats[self.__run] = copy.copy(self.best_score)
                self.__run += 1

                # get finish time
                now_fin = datetime.datetime.now()
                if self.time_to_search is not None and (now_fin - self.start_now).seconds > self.time_to_search:
                    if self.verbose > 0:
                        print("Random parameter search stopped early due to time overlapse")
                    self.__runok = False

                # return score or if early stopping has been initiated then return POOR score
                return {'loss': loss, 'params': params, 'status': STATUS_OK}
            except:
                return {'loss': 9999, 'params': params, 'status': STATUS_SUSPENDED}
        else:
            return {'loss': 9999, 'params': params, 'status': STATUS_SUSPENDED}

    def __reset(self):
        # When a new search is initiated then reset all the scores
        self.stats = {}
        self.__run = 0
        self.__init_score = cross_validation(self.est, self.__X, self.__y, cv=self.cv, scorer=self.scorer)
        self.best_score = self.__init_score

    def search(self):
        """
        Search the parameter grid
        """
        self.start_now = datetime.datetime.now()

        best = fmin(fn=self.__objective, space=self.__space, algo=self.__algo,
             max_evals=self.iters, trials=self.__trial, verbose=self.verbose,
             timeout=self.time_to_search)

        print("Starting HyperOpt Parameter Search")

        self.__reset()

        if self.best_score > self.__init_score:
            self.best_params = self.__trial.best_trial["result"]["params"]
        else:
            self.best_params = self.initparams
