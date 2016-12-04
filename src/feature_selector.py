
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor, GradientBoostingClassifier
import pandas as pd
from configs import *
from load_data import *



if __name__ == '__main__':
	# Define your classifier here
	# rfr = RandomForestRegressor(n_estimators = 310, n_jobs = -1, random_state = 2016, verbose = 1, max_features=13, min_samples_split=5)
	rfr = GradientBoostingClassifier(n_estimators = 35, random_state = 2016, verbose = 1, max_depth=6, min_samples_leaf=5)

	rfr.fit(train.values, np.ravel(train_target.values))
	
	col_names = train.columns.values

	importances = rfr.feature_importances_
	# std = np.std([tree.feature_importances_ for tree in rfr.estimators_],
	#              axis=0)

	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(train.shape[1]):
	    print("%d. %s (%f)" % (f + 1, col_names[indices[f]], importances[indices[f]]))