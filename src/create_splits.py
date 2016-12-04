from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from utilities import make_folder
from configs import *
import pickle as pkl
import pandas as pd

__author__ = 'abhinav'

paths = [PROCESS_PATH, FEATURES_PATH, DATASET_PATH, RESULTS_PATH]

for i in range(configs['n_folds']):
    paths.append(FOLD_PATH + 'fold' + str(i + 1))

for path in paths:
    make_folder(path)

y_train = pd.read_csv(INPUT_PATH + "train_target.csv", encoding="ISO-8859-1").values.ravel()
folds = list(StratifiedKFold(y_train, n_folds=configs['n_folds'], shuffle=True))
pkl.dump(folds, open(FOLD_PATH + 'folds.pkl', 'wb'))