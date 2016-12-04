from hyperopt import hp
from sklearn.ensemble import GradientBoostingClassifier
from settings import *
import numpy as np

gbcs = {
    'GradientBoosting': GradientBoostingClassifier(learning_rate= 0.03,
                                                   max_depth= 7,
                                                   max_features= 0.45,
                                                   n_estimators= 690,
                                                   min_samples_leaf= 14,
                                                   verbose=1,
                                                   random_state=configs['seed'])

}
h_param_grid = {
    "n_estimators": hp.quniform("n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 15, 1),
    "random_state": configs['seed'],
    "verbose": 1,
}
to_int_params = ['n_estimators', 'max_depth', 'min_samples_leaf']

level0 = gbcs['GradientBoosting']