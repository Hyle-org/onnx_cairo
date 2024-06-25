
from torch.utils.data import DataLoader

from giza.zkcook import serialize_model

## Load ferPLUS
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

import joblib
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx

import os
from ferplus import FerPlus

folder = '/home/alexandre/Documents/repos/facial-recog/FERPlus'
    
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

params = {
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 16,
    "colsample_bylevel": 0.7,
}

def categorical_model(X, y, output_dir):
    """Train using builtin categorical data support from XGBoost"""
    print("start training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1994, test_size=0.2
    )

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    # Specify `enable_categorical` to True.
    clf = xgb.XGBClassifier(
        **params,
        eval_metric="auc",
        enable_categorical=True,
        max_cat_to_onehot=1,  # We use optimal partitioning exclusively
    )
    
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)])
    clf.save_model(os.path.join(output_dir, "xgboost_model.json"))

    print(clf.get_params())

    # simplified_model, transformer = mcr(model = clf,
    #      X_train = X_train,
    #      y_train = y_train, 
    #      X_eval = X_test, 
    #      y_eval = y_test, 
    #      eval_metric = 'rmse',
    #      transform_features = True)
    
    #print(simplified_model.get_params())
    
    serialize_model(clf, "serialized.json")

    joblib.dump(clf, os.path.join(output_dir, "xgboost_model.pkl"))

    y_score = clf.predict_proba(X_test)[:, 1]  # proba of positive samples
    auc = roc_auc_score(y_test, y_score)
    print("AUC of using builtin categorical data support:", auc)
    return clf

model = categorical_model(img_dataframe, score_dataframe, '/home/alexandre/Documents/repos/facial-recog/')

initial_types = [('input', FloatTensorType([None, 48 * 48]))]
print(initial_types)

# Convertir le modèle XGBoost en format ONNX
onnx_model = convert_xgboost(model, initial_types=initial_types)
onnx.save_model(onnx_model, 'xgboost_model.onnx')