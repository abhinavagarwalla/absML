from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier
from settings import *

__author__ = 'abhinav'

# Random Forest Classifiers
rfrs = {
    'RandomForest': RandomForestClassifier(n_estimators=310,
                                           max_features=13,
                                           min_samples_split=6,
                                           n_jobs=-1,
                                           random_state=configs['seed'],
                                           verbose=1),
    'RandomForest2': RandomForestClassifier(n_estimators=310,
                                           max_features=30,
                                           min_samples_split=2,
                                           n_jobs=-1,
                                           random_state=configs['seed'],
                                           verbose=1),
    'RandomForest3': RandomForestClassifier(n_estimators=310,
                                           max_features=8,
                                           min_samples_split=6,
                                           n_jobs=-1,
                                           random_state=configs['seed'],
                                           verbose=1),
}
h_param_grid = {'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
                'max_features': hp.quniform('max_features', 5, 50, 1),
                'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1),
                }

to_int_params = ['n_estimators', 'max_features', 'min_samples_split']

level0 = rfrs['RandomForest']