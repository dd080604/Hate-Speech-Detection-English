
### How to load Features for Model Training:

```python
import numpy as np
import pickle

# Load feature extractor
with open("artifacts/feature_extractor.pkl", "rb") as f:
    fe = pickle.load(f)

# Load feature matrices
X_train = np.load("artifacts/X_train.npy")
X_val = np.load("artifacts/X_val.npy")
y_train = np.load("artifacts/y_train.npy")
y_val = np.load("artifacts/y_val.npy")

print("Loaded shapes:")
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
```
### Run this in the terminal to get the 'artifacts\X_train' file: 
```python
python -m feature_engineering.src.build_features
```