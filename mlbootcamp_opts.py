#!/usr/bin/env python

from sklearn.metrics import log_loss
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn_helpers import CategoricalTransformer, DummyEncoder

import numpy as np
import pandas as pd
import xgboost as xgb

X_train = pd.read_csv('files/x_train.csv', sep=';')
y_train = pd.read_csv('files/y_train.csv', sep=';', header=None)
X_train['averageLevelsPerDay'] = X_train['numberOfAttemptedLevels'] / X_train['numberOfDaysActuallyPlayed']
X_train['averageAttemptsPerLevel'] = (X_train['totalNumOfAttempts'] - X_train['attemptsOnTheHighestLevel']) / (X_train['numberOfAttemptedLevels'] - 1)
X_train['avarageScorePerLevel'] = X_train['totalScore'] / X_train['numberOfAttemptedLevels']
X_train['wastedBoosters'] = X_train['fractionOfUsefullBoosters'] * X_train['numberOfBoostersUsed']
X_train['userProgress'] = X_train['numberOfAttemptedLevels'] / ((X_train['maxPlayerLevel'] + 1)*X_train['numberOfDaysActuallyPlayed'])
X_train['attemptsRatio'] = X_train['attemptsOnTheHighestLevel'] / X_train['totalNumOfAttempts']
X_train['stuckOnHighest'] = X_train['attemptsOnTheHighestLevel'] / X_train['averageAttemptsPerLevel']
X_train['minPlayerLevel'] = X_train['maxPlayerLevel']-((X_train['totalNumOfAttempts'] - X_train['attemptsOnTheHighestLevel']))
X_train['playerLevelDelta'] = X_train['maxPlayerLevel'] - X_train['minPlayerLevel']
X_train['avarageScorePerAttempt'] = X_train['avarageScorePerLevel']/X_train['averageAttemptsPerLevel']
X_train['totalDays'] = (X_train['maxPlayerLevel'] - X_train['numberOfAttemptedLevels']) / X_train['averageLevelsPerDay']
X_train['averageScorePerDay'] = X_train['totalScore'] / X_train['numberOfDaysActuallyPlayed']
X_train['averageBonusScorePerDay'] = X_train['totalBonusScore']/X_train['numberOfDaysActuallyPlayed']
X_train['totalNumOfTurns'] = X_train['averageNumOfTurnsPerCompletedLevel']*X_train['numberOfAttemptedLevels']

X_train = X_train.fillna(0)
pipe = make_pipeline(CategoricalTransformer(['doReturnOnLowerLevels']), DummyEncoder(), StandardScaler())
X_train_trans = pipe.fit_transform(X_train)
y_train_trans = y_train[0].values

space = {
    'max_depth': hp.choice('max_depth', np.arange(0, 10, dtype=int)),
    'min_child_weight': hp.quniform('x_min_child', 1, 10, 1),
    'subsample': hp.uniform('x_subsample', 0.3, 1),
    'n_estimators': hp.choice('x_n_estimators', np.arange(10, 300, dtype=int)),
    'learning_rate': hp.uniform('x_learning_rate', 0.0001, 0.9)
}

space4svm = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20),
}

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,15)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
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

def objRf(params):
    clf = RandomForestClassifier(**params)
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
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(clf, X_train_trans, y_train_trans, scoring='neg_log_loss', cv=kfold)

    auc = abs(score.mean())
    print("SCORE: %0.7f" % auc)
    return {'loss': auc, 'status': STATUS_OK}


if __name__ == "__main__":
    trials = Trials()
    best = fmin( fn=objective, space=space, algo=tpe.suggest, max_evals=300, trials=trials)
    print(best)
