import numpy as np
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
from load_data import *


if __name__=="__main__":
	print('- Data Loaded')
	n_components=2
	# model = TSNE(n_components=2, random_state=0)
	model = MDS(n_components, max_iter=100, n_init=1)
	X1 = model.fit_transform(train[:10000].values) 
	plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=np.ravel(train_target.values[:10000]))
	plt.show()

