import pandas as pd
from configs import *

train = pd.read_table(INPUT_PATH + "train.csv", encoding="ISO-8859-1")
valid = pd.read_table(INPUT_PATH + "valid.csv", encoding="ISO-8859-1")
test = pd.read_table(INPUT_PATH + "test.csv", encoding="ISO-8859-1")

train_target = pd.read_table(INPUT_PATH + "train_target.csv", encoding="ISO-8859-1", header=None)
valid_target = pd.read_table(INPUT_PATH + "valid_target.csv", encoding="ISO-8859-1", header=None)

data_all = pd.concat([train, valid, test])
target_all = pd.concat([train_target, valid_target])
# TODO Change data type to float for corresponding values
# TODO Flatten target values
