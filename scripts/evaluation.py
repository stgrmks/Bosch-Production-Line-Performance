__author__ = 'MSteger'

import numpy as np
import pandas as pd
import os
from sklearn import metrics, model_selection, ensemble

def mcc_treshold(Y, Yhat, grid = 10):
    Yhat = Yhat[:, 1]
    thresholds = np.linspace(0.99, 0.99999, grid)
    mcc = np.array([metrics.matthews_corrcoef(Y, (Yhat > np.percentile(Yhat, t * 100)).astype(int)) for t in thresholds])
    return mcc.max()

if __name__ == '__main__':
    data_folder = r'/data/'
    seed = 1337
    folds = 2

    model = ensemble.RandomForestClassifier()
    X_train, y_train = pd.read_csv(os.path.join(data_folder, 'features_train.csv.gz'), index_col = 0), pd.read_csv(os.path.join(data_folder, 'Y.csv.gz'), usecols = ['Response'])['Response'].astype(np.int8)
    skf = model_selection.StratifiedKFold(n_splits = folds, shuffle = True, random_state = seed)
    scorer = metrics.make_scorer(score_func = mcc_treshold, greater_is_better = True, needs_proba = True)

    result = model_selection.cross_val_score(estimator = model, X = X_train, y = y_train, scoring = scorer, cv = skf, n_jobs = 1)

    print 'mean: {} std: {} performance: {}'.format(np.mean(result), np.std(result), result)
