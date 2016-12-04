from hyperopt import hp
from sklearn.ensemble import ExtraTreesClassifier

__author__ = 'abhinav'

etrees = {
    'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=350,
                                                 criterion='gini',
                                                 max_depth=None,
                                                 n_jobs=-1)
}

h_param_grid = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 1),
    'max_depth': hp.quniform('max_depth', 5, 50, 1)
}

to_int_params = ['n_estimators', 'max_depth']

level0 = etrees['ExtraTreesClassifier']