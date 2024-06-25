# Train the XGBoost model

```
python3 ./run_training.py
```

# Register model to Giza

```
giza transpile xgboost_model.onnx --output-path xgboost_onnx
```

# Deploy endpoint to make the model available
```
giza endpoints deploy --model-id 844 --version-id 6
```

# Call the model

Modify  `./run_call.py` accordingly
```
MODEL_ID = 844  # Update with your model ID
VERSION_ID = 6  # Update with your version ID
```
Then
```
python3 ./run_call.py
```