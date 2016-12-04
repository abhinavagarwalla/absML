from hyperopt import hp
from sklearn.tree import DecisionTreeClassifier
from settings import *
# class_weight = {0: 2, 1: 10, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 5, 12: 1,
#                 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1}

dtrees = {
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=29, random_state=configs['seed']),
    'DecisionTreeClassifier5': DecisionTreeClassifier(max_depth=38, random_state=configs['seed']),
    'DecisionTreeClassifier25': DecisionTreeClassifier(max_depth=25, random_state=configs['seed']),
    'DecisionTreeClassifier15': DecisionTreeClassifier(max_depth=15, random_state=configs['seed']),
    'DecisionTreeClassifier45': DecisionTreeClassifier(max_depth=45, random_state=configs['seed']),
    'DecisionTreeClassifier7': DecisionTreeClassifier(max_depth=26, random_state=configs['seed']),
}

dtree_space = {
    'max_depth': hp.quniform('max_depth', 5, 50, 1)
}
dtree_int = ['max_depth']
