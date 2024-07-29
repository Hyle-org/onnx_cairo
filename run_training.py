
from torch.utils.data import DataLoader

from giza.zkcook import serialize_model

## Load ferPLUS
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

import os
from ferplus import FerPlus

folder = './FERPlus'
    
training_data = FerPlus('Training', os.path.join(folder, "fer2013new.csv"), folder)
test_data = FerPlus('PublicTest', os.path.join(folder, "fer2013new.csv"), folder)

batch_size = 2**12

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

img_list = []
score_list= []

for X, y in training_data:
    img_list.append(X.numpy().flatten())
    score_list.append(y.numpy())
#    print(f"Shape of X [N, C, H, W]: {X.shape} {X.dtype}")
#    print(f"Shape of y: {y.shape} {y.dtype}")

print(len(img_list))
print(len(score_list))

img_dataframe = pd.DataFrame(img_list)
score_dataframe = pd.Series([0 if s < 0.3 else 1 for s in score_list])

params = {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.5, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 150, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1.0}

def categorical_model(X, y, model_file):
    """Train using builtin categorical data support from XGBoost"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1994, test_size=0.2
    )

    clf = xgb.XGBClassifier(
        **params,
        eval_metric="auc",
        enable_categorical=True,
        max_cat_to_onehot=1,  # We use optimal partitioning exclusively
    )
    
    clf.fit(X, y, eval_set=[(X_test, y_test), (X_train, y_train)])
    
    serialize_model(clf, model_file)

    y_score = clf.predict_proba(X_test)[:, 1]  # proba of positive samples
    print(y_score)
    auc = roc_auc_score(y_test, y_score)
    print("AUC of using builtin categorical data support:", auc)
    return clf
"""
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [300],
        'learning_rate': [0.5],
        'max_depth': [3],
        'min_child_weight': [3],
        'gamma': [0],
        'subsample': [1.0],
        'colsample_bytree': [0.8],
        'reg_alpha': [0],
        'reg_lambda': [1]
    }


    xgb_m = xgb.XGBClassifier()
    grid_search = GridSearchCV(estimator=xgb_m, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
"""

    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    # Specify `enable_categorical` to True.
    

categorical_model(img_dataframe, score_dataframe, "serialized_optimized.json")