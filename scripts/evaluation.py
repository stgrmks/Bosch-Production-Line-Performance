author = 'MSteger'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection

def create_cv(Y, folds=2):
    cv = list(model_selection.cross_validation.StratifiedKFold(Y, n_folds=folds, shuffle=True, random_state=seed))
    return cv

def split_data_cv(Y, x, cv, i):
    x_train, Y_train, x_test, Y_test = x.ix[cv[i][0]], Y[cv[i][0]], x.ix[cv[i][1]], Y[cv[i][1]]
    return Y_train, x_train, Y_test, x_test

def train_eval(model, Y_train, x_train, Y_test=None, x_test=None):
    model.fit(x_train, Y_train)
    if x_test is not None:
        Yhat = model.predict_proba(x_test)[:, 1]
        score = metrics.roc_auc_score(Y_test, Yhat)
        print 'out-of-sample metric', score
        return Yhat, score
    else:
        return model

def mcc_treshold(Y, Yhat, grid=10, plot=False):
    thresholds = np.linspace(0.99, 0.99999, grid)
    print 'training Yhat stats: '
    print 'Yhat min', Yhat.min(), 'Yhat max:', Yhat.max(), 'Mean', Yhat.mean()
    mcc = np.array([metrics.matthews_corrcoef(Y, (Yhat > np.percentile(Yhat, t * 100)).astype(int)) for t in thresholds])
    if plot:
        plt.plot(thresholds, mcc)
    best_threshold = thresholds[mcc.argmax()]
    print 'best threshold', best_threshold, 'gives MCC', mcc.max()
    return best_threshold, mcc.max()


def run_cv(Y, folds, params, grid=100, plot=False):
    cv = create_cv(Y, folds=folds)
    Yhat, score = np.ones(Y.shape), np.ones(folds)
    for i in range(folds):
        print 'fold', i + 1
        Yhat[cv[i][1]], score[i] = train_eval(xgb.XGBClassifier(**params), *split_data_cv(Y, x, cv, i))
    treshold, mcc = mcc_treshold(Y, Yhat, grid=grid, plot=plot)
    print 'mean out-of-sample score: ', score.mean(), '(+-', score.std(), ')'
    return Yhat, treshold, mcc, score.mean()

if __name__ == '__main__':
    seed = 1337
    skf = model_selection.
    print 'done'