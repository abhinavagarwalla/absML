INPUT_PATH = "../data/"

OUTPUT_PATH = "../output/"

FEATURES_PATH = "features/"

DATASET_PATH = 'datasets/'

RESULTS_PATH = 'results/'

configs = {
    'n_folds': 4,
    'seed': 2016,
    'silent': False
}

LEN_TRAIN = 761179
LEN_VALID = 1522358

TRAIN_FILE = INPUT_PATH + 'train.csv'
VALID_FILE = INPUT_PATH + 'valid.csv'
TEST_FILE = INPUT_PATH + 'test.csv'

dataset_name = 'main_normalized_log10'
model_name = 'valid'

DEBUG = False

if DEBUG:
    PROCESS_PATH = 'processing/'

    TEST_CUTOFF = 0.20
    mConfig = {
        'rfr_n_trees': 100,
        'etr_n_trees': 100,
        'gbr_n_trees': 100,
        'xgb_n_trees:linear': 263,
        'xgb_b_trees:logistic': 263
    }
    NJOBS = -1
else:
    PROCESS_PATH = 'processing/'
    TEST_CUTOFF = 0.20  # Calibrated with leaderboard +- 0.001
    mConfig = {
        'rfr_n_trees': 100,
        'etr_n_trees': 100,
        'gbr_n_trees': 100,
        'xgb_n_trees:linear': 263,
        'xgb_b_trees:logistic': 263
    }
    NJOBS = 2

FOLD_PATH = PROCESS_PATH + str(configs['n_folds']) + 'folds/'

float_formatter = lambda x: "%.3f" % x

# sklearn
skl_n_estimators_min = 100
skl_n_estimators_max = 1000
skl_n_estimators_step = 10

class_weight = {
    0: 2,
    1: 10,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 5,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1}



# class_ratio = [0.7572345007, 0.0006279732, 0.0139546677, 0.0214233446, 0.0302832842, 0.0402546576, 0.0577420029, 0.0147363498, 0.0068722337, 0.0019877059, 0.0062101030, 0.0003954392, 0.0010339224, 0.0019167633, 0.0010930412, 0.0004125179, 0.0007632896, 0.0018550170, 0.0257626655, 0.0154405206]
# s = 0
# class_weight={}

# for i in range(len(class_ratio)):
#     s += 1/class_ratio[i]

# for i in range(len(class_ratio)):
#     class_weight[i] = 1/(s*class_ratio[i])
