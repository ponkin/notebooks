#!/usr/bin/env python

from sklearn.metrics import log_loss
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import xgboost as xgb

X_train = pd.read_csv('files/x_train.csv', sep=';')
y_train = pd.read_csv('files/y_train.csv', sep=';', header=None)
pipe = make_pipeline(StandardScaler())
X_train_trans = pipe.fit_transform(X_train)
y_train_trans = y_train[0].values
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train_trans, y_train_trans, test_size=0.5, random_state=17)

space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'min_child_weight': hp.quniform('x_min_child', 1, 10, 1),
    'subsample': hp.uniform('x_subsample', 0.7, 1),
    'n_estimators': hp.choice('x_n_estimators', np.arange(800, 10000, dtype=int)),
    'learning_rate': hp.uniform('x_learning_rate', 0.001, 0.1)
}

spaceSvm = {
        'C': hp.quniform('C', 1, 10, 1),
        'gamma': hp.uniform('gamma', 0.01, 0.0001),
        'kernel': hp.choice('kernel', ['linear', 'rbf'])
        }

def objSVM(s):
    clf = SVC(
            cache_size=800,
            class_weight='balanced',
            C=s['C'],
            gamma=s['gamma'],
            kernel=s['kernel'],
            probability=True)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(clf, X_train_trans, y_train_trans, scoring='neg_log_loss', cv=kfold)
    loss = abs(score.mean())
    print("SCORE: %0.7f" % loss)
    return {'loss': loss, 'status': STATUS_OK}

def objective(space):

    clf = xgb.XGBClassifier(
        n_estimators=space['n_estimators'],
        learning_rate=space['learning_rate'],
        max_depth=space['max_depth'],
        min_child_weight=space['min_child_weight'],
        subsample=space['subsample'])

    eval_set = [(X_train1, y_train1), (X_test1, y_test1)]

    clf.fit(
        X_train1,
        y_train1,
        eval_set=eval_set,
        eval_metric="logloss",
        early_stopping_rounds=50)

    pred = clf.predict_proba(X_test1)[:, 1]
    auc = abs(log_loss(y_test1, pred))
    print("SCORE: %0.7f" % auc)
    return {'loss': auc, 'status': STATUS_OK}


if __name__ == "__main__":
    trials = Trials()
    best = fmin( fn=objSVM, space=spaceSvm, algo=tpe.suggest, max_evals=25, trials=trials)
    print(best)
