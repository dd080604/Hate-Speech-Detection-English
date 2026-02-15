import os
import pandas as pd
import numpy as np
import pickle


from hate_preproc.preprocess import preprocess_dataframe, add_signal_columns, PreprocessConfig
from feature_engineering.src.feature_extractor import FeatureExtractor
from feature_engineering.src.feature_config import get_feature_config, FEATURE_SETS, BASE_CONFIG
from sklearn.model_selection import train_test_split
from scipy import sparse

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_artifacts(out_dir: str, fe: FeatureExtractor, X_train, X_val, X_test, y_train, y_val):
    ensure_dir(out_dir)

    # Save extractor (vocab + idf + numeric stats)
    with open(os.path.join(out_dir, "feature_extractor.pkl"), "wb") as f:
        pickle.dump(fe, f)

    # Save matrices
    sparse.save_npz(os.path.join(out_dir, "X_train.npz"), sparse.csr_matrix(X_train))
    sparse.save_npz(os.path.join(out_dir, "X_val.npz"), sparse.csr_matrix(X_val))
    sparse.save_npz(os.path.join(out_dir, "X_test.npz"), sparse.csr_matrix(X_test))

    #np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    #np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    #np.save(os.path.join(out_dir, "X_test.npy"), X_test)

    # Save labels if you want them co-located
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)

def build_all_feature_sets(train_df, test_df=None):
    """
    Assumes train_df contains labels and text_clean + numeric columns already.
    You likely already have y extracted below—keep your current label logic.
    """

    # -------------------------
    # 1) split ONCE (consistent)
    # -------------------------
    label_col = "label"

    train_df, val_df = train_test_split(
        train_df,
        test_size = BASE_CONFIG["val_size"],
        random_state=BASE_CONFIG["split_seed"],
        stratify=train_df[label_col]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if test_df is not None:
        test_df = test_df.reset_index(drop=True)

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    # ===============================
    # Add Signal Columns    

    pp_cfg = PreprocessConfig()

    train_df = add_signal_columns(train_df, pp_cfg)
    val_df = add_signal_columns(val_df, pp_cfg)
    if test_df is not None:
        test_df = add_signal_columns(test_df, pp_cfg)


    # ===============================
    # Clean Text

    train_df = preprocess_dataframe(train_df, pp_cfg, out_col="text_clean")
    val_df = preprocess_dataframe(val_df, pp_cfg, out_col="text_clean")
    if test_df is not None:
        test_df = preprocess_dataframe(test_df, pp_cfg, out_col ="text_clean")
    # --- YOU: keep your current code here ---
    # train_split_df = ...
    # val_split_df = ...
    # y_train = ...
    # y_val = ...
    # If you also have an official test set:
    # test_df = ...

    # ------------------------------------------------
    # 2) loop through each feature set and build/save
    # ------------------------------------------------
    for feature_name in FEATURE_SETS.keys():
        feat_cfg = get_feature_config(feature_name)

        out_dir = os.path.join("artifacts", feature_name)
        ensure_dir(out_dir)

        fe = FeatureExtractor(config=feat_cfg)

        X_train = fe.fit_transform(train_df)
        X_val = fe.transform(val_df)
        if test_df is not None:
            X_test = fe.transform(test_df)
        else:
            X_test = np.zeros((0, X_train.shape[1]), dtype=float)


        # Save
        save_artifacts(
            out_dir=out_dir,
            fe=fe,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
        )

        print(f"✅ Saved {feature_name} -> {out_dir}")


if __name__ == "__main__":
    train_df = pd.read_csv("hate_speech_train.csv")
    test_df = pd.read_csv("hate_speech_test.csv")
    build_all_feature_sets(train_df, test_df=test_df)




#feature_name = FEATURE_CONFIG["feature_set_name"]
#out_dir = f"artifacts/{feature_name}"

#cfg = PreprocessConfig()


#train_df_full = pd.read_csv("hate_speech_train.csv")
#test_df = pd.read_csv("hate_speech_test.csv")

#label_col = "label"

#train_df, val_df = train_test_split(
#    train_df_full,
#    test_size = FEATURE_CONFIG["val_size"],
#    random_state=FEATURE_CONFIG["split_seed"],
#    stratify=train_df_full[label_col]
#)

#train_df = train_df.reset_index(drop=True)
#val_df = val_df.reset_index(drop=True)
#test_df = test_df.reset_index(drop=True)

# ===============================
# Add Signal Columns

#train_df = add_signal_columns(train_df, cfg)
#val_df = add_signal_columns(val_df, cfg)
#test_df = add_signal_columns(test_df, cfg)


# ===============================
# Clean Text

#train_df = preprocess_dataframe(train_df, cfg, out_col="text_clean")
#val_df = preprocess_dataframe(val_df, cfg, out_col="text_clean")
#test_df = preprocess_dataframe(test_df, cfg, out_col ="text_clean")

#### #### print(train_df.head())

# ===============================
# Feature Extractor
#fe = FeatureExtractor()

## TODO!! Validation Set

#X_train = fe.fit_transform(train_df)
#X_val = fe.transform(val_df)
#X_test = fe.transform(test_df)

#y_train = train_df["label"].to_numpy()
#y_val = val_df["label"].to_numpy()

#print("X_train Shape:", X_train.shape)
#print("X_val shape:", X_val.shape)

#print(X_train.shape, X_val.shape, X_test.shape)
#print((X_train[:, :fe.vocab_size].sum(axis=1) == 0).mean())
#print(sum("_" in tok for tok in fe.word_vocab))

#import numpy as np
#print("NaNs train:", np.isnan(X_train).sum(), "Infs train:", np.isinf(X_train).sum())
#print("Nans val:" !! complete

## ## ### check bigrams ### ## ##
##print(len([tok for tok in fe.word_vocab.keys() if "_" in tok]))
##print([tok for tok in fe.word_vocab.keys() if "_" in tok][:16])

### ================================
### ### Save feature extractor

#with open("{out_dir}/feature_extractor.pickle", "wb") as f:
#    pickle.dump(fe, f)

## feature matrices

#np.save("{outdir}/X_train.npy", X_train)
#np.save("{outdir}/y_train.npy", y_train)
#np.save("{outdir}/X_test.npy", X_test)
#np.save("{outdir}/X_val.npy", X_val)
#np.save("{outdir}/y_val.npy", y_val)


