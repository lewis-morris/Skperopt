# Skperopt
A hyperopt wrapper - making it easy to use with Sklearn.

Just pass in an estimator and a parameter grid and your good to go.

Its super easy to use Skperopt - 

1. Import skperopt
2. Initalize hyperopt 
3. Run skperopt.HyperSearch
4. Collect the results

Code example below.

'''python
import pandas as pd
import numpy as np
import skperopt as sk
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

#generate classification data
data = make_classification(n_samples=1000, n_features=10, n_classes=2)
X = pd.DataFrame(data[0])
y = pd.DataFrame(data[1])

#init the classifier
kn = KNeighborsClassifier()
param = {"n_neighbors": [int(x) for x in np.linspace(1, 300, 30)],
         "leaf_size": [int(x) for x in np.linspace(1, 200, 30)],
         "p": [1, 2, 3, 4, 5, 10, 20],
         "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
         "weights": ["uniform", "distance"]}


#search parameters
search = sk.HyperSearch(kn, X, y, 10000, 20, cv=5, scorer="f1", verbose=2, params=param)
search.search()

#apply best parameters
kn.set_params(**search.best_params)'''
