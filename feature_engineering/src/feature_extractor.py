import numpy as np
from collections import Counter

from .feature_config import FEATURE_CONFIG
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

        txt_col = "text_clean"
        texts = df[txt_col].fillna("").astype(str).tolist()

        # word vocab and idf
        if self.config["use_word_features"]:
            self._build_word_vocab(texts)
            if self.config["tfidf"]:
                self.compute_idf(texts)

        # numeric feature stats
        num_cols = self.config["numeric_feature_cols"]
        if num_cols:
            X_num = df[num_cols].fillna(0).to_numpy(dtype=float)
            self.numeric_dim = X_num.shape[1]

            if self.config["normalize_features"]:
                self.numeric_means = X_num.mean(axis=0)
                self.numeric_std = X_num.std(axis=0)
                self.numeric_std = np.where(self.numeric_std == 0, 1.0, self.numeric_std)
            else:
                self.numeric_means = None
                self.numeric_std = None

        if self.config["verbose_features"]:
            total_dim = self.vocab_size + self.numeric_dim
            print(f"word dim: {self.vocab_size},\n numeric dim: {self.numeric_dim}, \n total dim: {total_dim}")

    def transform(self, df):

        text_col = "text_clean"
        texts = df[text_col].fillna("").astype(str).tolist()

        if self.config["use_word_features"]:
            X_word = np.vstack([self._word_vector(t) for t in texts])
        else:
            X_word = np.zeros((len(texts), 0), dtype=float)

        num_cols = self.config["numeric_feature_cols"]
        if num_cols:
                X_num = df[num_cols].fillna(0).to_numpy(dtype=float)
                if self.config["normalize_features"]:
                    X_num = (X_num - self.numeric_means) / self.numeric_std
        else:
            X_num = np.zeros((len(texts), 0), dtype=float)

        return np.concatenate([X_word, X_num], axis=1)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_bigrams(self, words):
        return [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]

    def _build_word_vocab(self, texts):
        counter = Counter()
        doc_counter = Counter()

        for txt in texts:
            words = txt.split()
            tokens = list(words)

            if self.config["bigrams"]:
                bigrams = self.get_bigrams(words)
                tokens.extend(bigrams)

            counter.update(tokens)
            doc_counter.update(set(tokens))

        min_freq = self.config["min_word_freq"]
        max_ratio = self.config["common_word_ratio"]
        total_length = len(texts)

        candidates = []

        for token, freq in counter.items():
            if freq < min_freq:
                continue
            if (doc_counter[token] / total_length) > max_ratio:
                continue

            candidates.append((token, freq))

        if self.config["max_vocab_size"] is not None:
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:self.config["max_vocab_size"]]

        self.word_vocab = {token:i for i, (token, _) in enumerate(candidates)}
        self.vocab_size = len(self.word_vocab)

        if self.config["verbose_vocab"]:
            print(f"vocab size: {self.vocab_size}")

    def compute_idf(self, texts):
        counts = np.zeros(self.vocab_size, dtype=float)
        total_length = len(texts)
        eps = self.config["epsilon"]

        for txt in texts:
            words = txt.split()
            tokens = list(words)
            if self.config["bigrams"]:
                tokens.extend(self.get_bigrams(words))

            for tok in set(tokens):
                idx = self.word_vocab.get(tok)
                if idx is not None:
                    counts[idx] += 1

        self.idf = np.log((total_length + 1) / (counts + 1 + eps)) + 1



    def _word_vector(self, text):
        vec = np.zeros(self.vocab_size, dtype=float)
        words = text.split()

        tokens = list(words)

        if self.config["bigrams"]:
            tokens.extend(self.get_bigrams(words))

        for token in tokens:
            idx = self.word_vocab.get(token)
            ### bugfix is NOT --> is None
            if idx is None:
                continue
            if self.config["binary_cts"]:
                vec[idx] = 1
            else :
                vec[idx] += 1

        if self.config["tfidf"] and self.idf is not None:
            denominator = vec.sum() + self.config["epsilon"]
            tf = vec / denominator
            vec = tf * self.idf

        return vec


