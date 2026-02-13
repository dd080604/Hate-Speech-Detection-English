import numpy as np
from collections import Counter

from feature_config import FEATURE_CONFIG
from hate_preproc.src.hate_preproc.preprocess import normalize_text


class FeatureExtractor:
    def __init__(self, config=FEATURE_CONFIG):
        self.config = config

        self.word_vocab = {}
        self.idf = None
        self.vocab_size = 0

        self.numeric_means = None
        self.numeric_std = None
        self.numeric_dim = 0

    def fit(self, df):

        txt_col = self.config["text_clean"]
        texts = df[txt_col].fillna("").astype(str).tolist()

        # word vocab and idf
        if self.config["use_word_features"]:
            self.build_word_vocab(texts)
            if self.config["use_tfidf"]:
                self.compute_idf(texts)

        # numeric feature stats
        num_cols = self.config["numeric_feature_cols"]
        if num_cols:
            X_num = df[num_cols].fillna(0).to_numpy(dtype=float)
            sef.numeric_dim = X_num.shape[1]

            if self.config["normalize_features"]:
                self.numeric_means = X_num.mean(axis=0)
                self.numeric_stds = X_num.std(axis=0)
                self.numeric_stds = np.where(self.numeric_stds == 0, 1.0, self.numeric_stds)
            else:
                self.numeric_means = None
                self.numeric_stds = None

        if self.config["verbose_features"]:
            total_dim = self.vocab_size + self.numeric_dim
            print(f"word dim: {self.vocab_size},\n numeric dim: {self.numeric_dim}, \n total dim: {total_dim}")

    def transform(self, df):

        text_col = self.config["text_clean"]
        texts = df[text_col].fillna("").astype(str).tolist()

        if self.config["use_word_features"]:
            X_word = np.vstack([self._word_vector(t) for t in texts])
        else:
            X_word = np.zeros((len(texts), 0), dtype=float)

        num_cols = self.config["numeric_feature_cols"]
        if num_cols:
                X_num = df[num_cols].fillna(0).to_numpy(dtype=float)
                if self.config["normalize_features"]:
                    X_num = (X_num - self.numeric_means) / self.numeric_stds
        else:
            X_num = np.zeros((len(texts), 0), dtype=float)

        return np.concatenate([X_word, X_num], axis=1)

    def fit_transform():
        return

    def _build_word_vocab(self, texts):
        counter = Counter()
        ...
        return

    def compute_idf(self, texts):
        return

    def _word_vector(self, text):
        return vec