# Skperopt
A hyperopt wrapper - making it easy to use with Sklearn.

Just pass in an estimator and a parameter grid and your good to go.

1. Import skperopt
2. Initalize skperopt 
3. Run skperopt.HyperSearch.search
4. Collect the results

Code example below.

```python
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
param = {"n_neighbors": [int(x) for x in np.linspace(1, 60, 30)],
         "leaf_size": [int(x) for x in np.linspace(1, 60, 30)],
         "p": [1, 2, 3, 4, 5, 10, 20],
         "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
         "weights": ["uniform", "distance"]}


#search parameters
search = sk.HyperSearch(kn, X, y, params=param)
search.search()

#apply best parameters
kn.set_params(**search.best_params)

```

## HyperSearch Parameters

* **est** (*[sklearn estimator]*) 
         any sklearn style estimator

* **X** (*[pandas Dataframe]*) - your training data

* **y** (*[pandas Dataframe]*) - your training data

* **iters** (default 500 *[int]*) - number of iterations to try before early stopping

* **time_to_search** (default None *[int]*) - time in seconds to run for before early stopping (None = no time limit)

* **cv** (default 5 *[int]*) - number of folds to use in cross_vaidation tests

* **scorer** (default "f1" *[str]*) - type of evaluation metric to use - accepts "f1","auc","accuracy" or "rmse"

* **verbose** (default 1 *[int]*) - amount of verbosity 0 = none 1 = some 2 = debug

* **random** (default - *False*) - should the data be randomized during the cross validation

* **foldtype** (default "Kfold" *[str]*) - type of folds to use - accepts "KFold", "Stratified"

