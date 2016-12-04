from configs import *
from utilities import custom_scoring
from models.stacking import models
import pickle as pkl
import numpy as np

import os
import sys

clf_name = sys.argv[1]
dataset_name_new = dataset_name if len(sys.argv) < 3 else sys.argv[2]
level = 0 if len(sys.argv) < 4 else sys.argv[2]

clf = models[clf_name]

X_train = np.load('%s%s_train.npy' % (DATASET_PATH, dataset_name_new))
X_valid = np.load('%s%s_valid.npy' % (DATASET_PATH, dataset_name_new))
y_train = np.load('%sY_train.npy' % DATASET_PATH)
y_valid = np.load('%sY_valid.npy' % DATASET_PATH)

n_folds = configs['n_folds']  # Higher is better, not necessary

skf = pkl.load(open(FOLD_PATH + 'folds.pkl', 'rb'))
n_classes = len(set(y_train))

stack_train = np.zeros((X_train.shape[0]), dtype=int)
stack_valid = np.zeros((X_valid.shape[0]), dtype=int)

stack_train_prob = np.zeros((X_train.shape[0], n_classes))
stack_valid_prob = np.zeros((X_valid.shape[0], n_classes))


print('X_valid.shape = %s' % (str(X_valid.shape)))
print('stack_train.shape = %s' % (str(stack_train.shape)))
print('stack_valid.shape = %s' % (str(stack_valid.shape)))

if os.path.isfile('%s%d_fold_%s_%s.npy' % (FEATURES_PATH, configs['n_folds'], dataset_name_new, clf_name)):
    print('Training Classifier [%s] Done' % (clf_name))
    stack_all = np.load('%s%d_fold_%s_%s.npy' % (FEATURES_PATH, configs['n_folds'], dataset_name_new, clf_name))
    stack_all_prob = np.load('%s%d_fold_%s_%s_prob.npy' % (FEATURES_PATH, configs['n_folds'], dataset_name_new, clf_name))
else:
    print('Training classifier [%s]' % (clf_name))
    for i, (train_index, cv_index) in enumerate(skf):
        print('Fold [%s]' % (i))

        # This is the training and validation set
        X_train_temp = X_train[train_index]
        y_train_temp = y_train[train_index]
        X_cv = X_train[cv_index]
        Y_cv = y_train[cv_index]

        clf.fit(X_train_temp, y_train_temp)

        stack_train[cv_index] = clf.predict(X_cv)
        stack_train_prob[cv_index, :] = clf.predict_proba(X_cv)

    # Training the final classifier
    print('Training final classifier...')
    clf.fit(X_train, y_train)
    stack_valid[:] = clf.predict(X_valid)
    stack_valid_prob[:, :] = clf.predict_proba(X_valid)

    stack_all = np.concatenate([stack_train, stack_valid])
    stack_all.dump('%s%d_fold_%s_%s.npy' % (FEATURES_PATH, configs['n_folds'], dataset_name_new, clf_name))

    stack_all_prob = np.concatenate([stack_train_prob, stack_valid_prob])
    stack_all_prob.dump('%s%d_fold_%s_%s_prob.npy' % (FEATURES_PATH, configs['n_folds'], dataset_name_new, clf_name))

    clf.fit(X_train, y_train)

print('Scoring...')
print('Cross-Validation Score:...%5.5f' % custom_scoring(stack_all[:LEN_TRAIN], y_train))
print('Validation Score:.........%5.5f' % custom_scoring(stack_all[LEN_TRAIN:LEN_VALID], y_valid))
