import pandas as pd
import numpy as np
import pickle


from hate_preproc.preprocess import preprocess_dataframe, add_signal_columns, PreprocessConfig
from feature_engineering.src.feature_extractor import FeatureExtractor
from .feature_config import FEATURE_CONFIG
from sklearn.model_selection import train_test_split

cfg = PreprocessConfig()


train_df_full = pd.read_csv("hate_speech_train.csv")
test_df = pd.read_csv("hate_speech_test.csv")

label_col = "label"

train_df, val_df = train_test_split(
    train_df_full,
    test_size = FEATURE_CONFIG["val_size"],
    random_state=FEATURE_CONFIG["split_seed"],
    stratify=train_df_full[label_col]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ===============================
# Add Signal Columns

train_df = add_signal_columns(train_df, cfg)
val_df = add_signal_columns(val_df, cfg)
test_df = add_signal_columns(test_df, cfg)


# ===============================
# Clean Text

train_df = preprocess_dataframe(train_df, cfg, out_col="text_clean")
val_df = preprocess_dataframe(val_df, cfg, out_col="text_clean")
test_df = preprocess_dataframe(test_df, cfg, out_col ="text_clean")

#### #### print(train_df.head())

# ===============================
# Feature Extractor
fe = FeatureExtractor()

## TODO!! Validation Set

X_train = fe.fit_transform(train_df)
X_val = fe.transform(val_df)
X_test = fe.transform(test_df)

y_train = train_df["label"].to_numpy()
y_val = val_df["label"].to_numpy()

#print("X_train Shape:", X_train.shape)
#print("X_val shape:", X_val.shape)

print(X_train.shape, X_val.shape, X_test.shape)
print((X_train[:, :fe.vocab_size].sum(axis=1) == 0).mean())
print(sum("_" in tok for tok in fe.word_vocab))

import numpy as np
#print("NaNs train:", np.isnan(X_train).sum(), "Infs train:", np.isinf(X_train).sum())
#print("Nans val:" !! complete

## ## ### check bigrams ### ## ##
##print(len([tok for tok in fe.word_vocab.keys() if "_" in tok]))
##print([tok for tok in fe.word_vocab.keys() if "_" in tok][:16])

### ================================
### ### Save feature extractor

with open("artifacts/word12/feature_extractor.pickle", "wb") as f:
    pickle.dump(fe, f)

## feature matrices

np.save("artifacts/word12/X_train.npy", X_train)
np.save("artifacts/word12/y_train.npy", y_train)
np.save("artifacts/word12/X_test.npy", X_test)
np.save("artifacts/word12/X_val.npy", X_val)
np.save("artifacts/word12/y_val.npy", y_val)


