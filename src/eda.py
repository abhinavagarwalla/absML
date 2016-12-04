# Exploratory Data Analysis
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor, GradientBoostingClassifier
import pandas as pd
from configs import *
from load_data import *

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif, chi2


if __name__=="__main__":
	X = train.values
	y = train_target.values.ravel()
	print('- Data Loaded')

	plt.figure(1)
	plt.clf()

	X_indices = np.arange(X.shape[-1])
	X_labels = train.columns.values

	###############################################################################
	# Univariate feature selection with F-test for feature scoring
	# We use the default selection function: the 10% most significant features
	selector = SelectPercentile(f_classif, percentile=10)
	selector.fit(X, y)
	scores = -np.log10(selector.pvalues_)

	plt.bar(X_indices , scores, width=.2,
	        label=r'Univariate score ($-Log(p_{value})$)', color='g')

	plt.title("Comparing feature selection")
	plt.xlabel('Feature number')
	plt.xticks(X_indices, X_labels, rotation=90)

	plt.axis('tight')
	plt.legend(loc='upper right')
	plt.show()
