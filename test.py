import pandas as pd
import numpy as np
from skiperopt import search

from sklearn.datasets import make_classification

data = make_classification(n_samples=1000,n_features=10,n_classes=2)
X = pd.DataFrame(data[0])
y = pd.DataFrame(data[1])

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
param = {"n_neighbors": [int(x) for x in np.linspace(1, 300, 30)],
         "leaf_size": [int(x) for x in np.linspace(1, 200, 30)],
         "p": [1, 2, 3, 4, 5, 10, 20],
         "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
         "weights": ["uniform", "distance"]}

search = search.HyperSearch(kn,X,y,10000,20,cv=5,scorer="f1",verbose=1,params=param)
search.search()
kn.set_params(**search.best_params)