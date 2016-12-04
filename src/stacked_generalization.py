"""

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from models.stacking import level0
np.random.seed(2016)

from configs import *
from utilities import *
import pickle as pkl

np.set_printoptions(formatter={'float_kind': float_formatter})


def run(X, Y, X_test=None, Y_test=None):

    clfs = level0

    # Pre-allocate the data
    blend_train = np.zeros((X.shape[0], len(clfs)), dtype=int)  # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs)), dtype=int)  # Number of testing data x Number of classifiers

    
    print('blend_train.shape = %s' % (str(blend_train.shape)))
    print('blend_test.shape = %s' % (str(blend_test.shape)))

    # For each classifier, we train the number of fold times (=len(skf))
    for j, (clf_name, clf) in enumerate(clfs):
        blend_test_j = np.zeros((X_test.shape[0], len(
            skf)), dtype=int)  # Number of testing data x Number of folds , we will take the mean of the predictions later
        if os.path.isfile('%s%s%s' % (FOLD_PATH_NEW, clf_name, 'Train.npy')) and os.path.isfile(
                        '%s%s%s' % (FOLD_PATH_NEW, clf_name, 'Test.npy')):
            print('Loading classifier [%s %s]' % (j, clf_name))
            blend_train[:, j] = np.load(FOLD_PATH_NEW + clf_name + 'Train.npy')
            blend_test[:, j] = np.load(FOLD_PATH_NEW + clf_name + 'Test.npy')
        else:
            print('Training classifier [%s %s]' % (j, clf_name))
            for i, (train_index, cv_index) in enumerate(skf):
                print('Fold [%s]' % (i))

                # This is the training and validation set
                X_train = X[train_index]
                Y_train = Y[train_index]
                X_cv = X[cv_index]
                Y_cv = Y[cv_index]

                clf.fit(X_train, Y_train)

                # This output will be the basis for our blended classifier to train against,
                # which is also the output of our classifiers
                blend_train[cv_index, j] = clf.predict(X_cv)
                blend_test_j[:, i] = clf.predict(X_test)
            # Take the mean of the predictions of the cross validation set
            blend_test[:, j] = blend_test_j.mean(1)
            if not DEBUG or 1:
                blend_train[:, j].dump(FOLD_PATH_NEW + clf_name + 'Train.npy')
                blend_test[:, j].dump(FOLD_PATH_NEW + clf_name + 'Test.npy')

    print('Y.shape = %s' % Y.shape)

    # Saving Model Data
    blend_train.dump(FOLD_PATH_NEW + 'BlendTrain_X.npy')
    Y.dump(FOLD_PATH_NEW + 'BlendTrain_Y.npy')
    blend_test.dump(FOLD_PATH_NEW + 'BlendTest_X.npy')
    if 'Y_test' in locals(): Y_test.dump(FOLD_PATH_NEW + 'BlendTest_Y.npy')

    # Correlation Matrix
    print('\n---------- Correlation Matrix ----------')
    print(np.corrcoef(np.transpose(blend_train)))

    # Start blending!
    # bclf = LinearRegression()
    bclf = RandomForestClassifier()
    
    bclf.fit(blend_train, Y)

    # Predict now
    Y_test_predict = bclf.predict(blend_test)

    
    if 'Y_test' in locals():
        print('\n---------- Test Accuracy ----------')
        for i, (clf_name, clf) in enumerate(clfs):
            score = custom_scoring(blend_test[:, i], Y_test)
            print('%s Accuracy = %s' % (clf_name, score))
        score = custom_scoring(Y_test, Y_test_predict)
        print('Accuracy = %s' % (score))

    print('\n---------- Cross Validation Accuracy ----------')
    for i, (clf_name, clf) in enumerate(clfs):
        score = custom_scoring(blend_train[:, i], Y)
        print('%s Accuracy = %s' % (clf_name, score))
    # print('Weights = %s' % str(bclf.coef_))
    return Y_test_predict


def run_tests(X_train, y_train):
    pass


if __name__ == '__main__':
    # dataset_name = 'main'
    # model_name = 'testing1'
    if DEBUG:
        model_name = 'test_' + model_name
    FOLD_PATH_NEW = FOLD_PATH + model_name + '/'
    make_folder(FOLD_PATH_NEW)

    X_train = np.load('%s%s_train.npy' % (DATASET_PATH, dataset_name))
    X_valid = np.load('%s%s_valid.npy' % (DATASET_PATH, dataset_name))
    y_train = np.load('%sY_train.npy' % DATASET_PATH).ravel()
    y_valid = np.load('%sY_valid.npy' % DATASET_PATH).ravel()

    Y_test = run(X_train, y_train, X_valid, y_valid)
    # if not DEBUG:
    #     pd.DataFrame({"id": id_test, "relevance": Y_test}).to_csv('submission/submission_stacked_%s.csv' % time.time(),
    #                                                               index=False)