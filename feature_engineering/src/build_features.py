import pandas as pd
import numpy as np
import pickle


from hate_preproc.src.hate_preproc.preprocess import preprocess_dataframe, add_signal_columns, PreprocessConfig
from feature_engineering.src.feature_extractor import FeatureExtractor

cfg = PreprocessConfig()


train_df = pd.read_csv("hate_speech_train.csv")
#val_df = pd.read_csv("data/val.csv")

# ===============================
# Add Signal Columns

train_df = add_signal_columns(train_df, cfg)
#val_df = add_signal_columns(val_df, cfg)


# ===============================
# Clean Text

train_df = preprocess_dataframe(train_df, cfg, out_col="text_clean")
#val_df = preprocess_dataframe(val_df, cfg, out_col="text_clean")

#### #### print(train_df.head())

# ===============================
# Feature Extractor
fe = FeatureExtractor()

## TODO!! Validation Set

X_train = fe.fit_transform(train_df)
#X_val = fe.transform(val_df)
y_train = train_df["label"].to_numpy()
#y_val = val_df["label"].to_numpy()

print("X_train Shape:", X_train.shape)
#print("X_val shape:", X_val.shape)

#print("word_vocab/n", sorted(fe.word_vocab.items(), key=lambda x: x[1], reverse=True)[0:21])

import numpy as np
### ================================
### ### Save feature extractor

with open("artifacts/feature_extractor.pickle", "wb") as f:
    pickle.dump(fe, f)

## feature matrices

np.save("artifacts/X_train.npy", X_train)
np.save("artifacts/y_train.npy", y_train)
#np.save("artifacts/X_val.npy", X_val)
#np.save(""artifacts/y_val.npy", y_val)