import numpy as np
from collections import Counter

from .feature_config import FEATURE_CONFIG


class FeatureExtractor:
    def __init__(self, config=FEATURE_CONFIG):
        self.config = config

        self.word_vocab = {}
        self.idf = None
        self.vocab_size = 0

        self.char_vocab = {}
        self.char_idf = None
        self.char_vocab_size = 0

        self.numeric_means = None
        self.numeric_stds = None
        self.numeric_dim = 0
 

    def fit(self, df):

        txt_col = "text_clean"
        texts = df[txt_col].fillna("").astype(str).tolist()

        # word vocab and idf
        if self.config.get("use_word_features", False):
            self._build_word_vocab(texts)
            if self.config.get("tfidf_word", False):
                self.compute_word_idf(texts)

        # char vocab and idf
        if self.config.get("use_char_features", False):
            self._build_char_vocab(texts)
            if self.config.get("tfidf_char", False):
                self.compute_char_idf(texts)

        # numeric feature stats
        num_cols = self.config.get("numeric_feature_cols", [])
        if num_cols:
            X_num = df[num_cols].fillna(0).to_numpy(dtype=float)
            self.numeric_dim = X_num.shape[1]

            if self.config.get("normalize_features", True):
                self.numeric_means = X_num.mean(axis=0)
                self.numeric_stds = X_num.std(axis=0)
                self.numeric_stds = np.where(self.numeric_stds == 0, 1.0, self.numeric_stds)
            else:
                self.numeric_means = None
                self.numeric_stds = None
                self.numeric_dim = 0

        if self.config.get("verbose_features", False):
            total_dim = self.vocab_size + self.char_vocab_size + self.numeric_dim
            print(
                f"word dim: {self.vocab_size},\n"
                f"char dim: {self.char_vocab_size},\n"
                f"numeric dim: {self.numeric_dim},\n"
                f"total dim: {total_dim}"
                )

    def transform(self, df):

        text_col = "text_clean"
        texts = df[text_col].fillna("").astype(str).tolist()
        n = len(texts)

        # word matrix
        if self.config.get("use_word_features", False):
            X_word = np.vstack([self._word_vector(t) for t in texts])
        else:
            X_word = np.zeros((n, 0), dtype=float)

        # char matrix
        if self.config.get("use_char_features", False):
            X_char = np.vstack([self._char_vector(t) for t in texts])
        else:
            X_char = np.zeros((n, 0), dtype=float)    

        # numeric matrix
        num_cols = self.config.get("numeric_feature_cols", [])
        if num_cols:
                X_num = df[num_cols].fillna(0).to_numpy(dtype=float)
                if self.config.get("normalize_features", True) and self.numeric_means is not None:
                    X_num = (X_num - self.numeric_means) / self.numeric_stds
        else:
            X_num = np.zeros((n, 0), dtype=float)

        return np.concatenate([X_word, X_char, X_num], axis=1)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    # ------------------- word features ----------------------------
    def get_bigrams(self, words):
        return [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]

    def _build_word_vocab(self, texts):
        counter = Counter()
        doc_counter = Counter()

        use_bigrams = self.config.get("bigrams", False)

        for txt in texts:
            words = txt.split()
            tokens = list(words)

            if use_bigrams:
                bigrams = self.get_bigrams(words)
                tokens.extend(bigrams)

            counter.update(tokens)
            doc_counter.update(set(tokens))

        min_freq = int(self.config.get("min_word_freq", 1))
        max_ratio = float(self.config.get("common_word_ratio", 1.0))
        max_vocab_size = self.config.get("max_vocab_size", None)
        total_docs = len(texts)

        candidates = []

        for token, freq in counter.items():
            if freq < min_freq:
                continue
            if total_docs > 0 and (doc_counter[token] / total_docs) > max_ratio:
                continue

            candidates.append((token, freq))

        if max_vocab_size is not None:
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[: int(max_vocab_size)]

        self.word_vocab = {token:i for i, (token, _) in enumerate(candidates)}
        self.vocab_size = len(self.word_vocab)

        if self.config.get("verbose_vocab", False):
            print(f"vocab size: {self.vocab_size}")

    def compute_word_idf(self, texts):
        counts = np.zeros(self.vocab_size, dtype=float)
        total_docs = len(texts)
        eps = float(self.config.get("epsilon", 1e-6))
        use_bigrams = self.config.get("bigrams", False)

        for txt in texts:
            words = txt.split()
            tokens = set(words)
            if use_bigrams:
                tokens.update(self.get_bigrams(words))

            for tok in tokens:
                idx = self.word_vocab.get(tok)
                if idx is not None:
                    counts[idx] += 1

        self.idf = np.log((total_docs + 1) / (counts + 1 + eps)) + 1



    def _word_vector(self, text):
        vec = np.zeros(self.vocab_size, dtype=float)
        words = text.split()

        tokens = list(words)

        if self.config.get("bigrams", False):
            tokens.extend(self.get_bigrams(words))

        binary = self.config.get("binary_cts_word", False)
        for token in tokens:
            idx = self.word_vocab.get(token)
            if idx is None:
                continue
            if binary:
                vec[idx] = 1
            else :
                vec[idx] += 1

        if self.config.get("tfidf_word", False) and self.idf is not None:
            eps = float(self.config.get("epsilon", 1e-6))
            denominator = vec.sum() + eps
            tf = vec / denominator
            vec = tf * self.idf
            norm = np.linalg.norm(vec) + eps
            vec = vec / norm

        return vec

    # ----------------- char features -----------------
    def _char_ngrams(self, text):
        """
        Return list of character n-grams for n in [char_n_min, char_n_max].
        Pads with spaces so boundary info can be learned.
        """
        n_min = int(self.config.get("char_n_min", 3))
        n_max = int(self.config.get("char_n_max", 5))
        # text_clean should already be normalized; keep it simple
        t = f" {text} "
        grams = []
        L = len(t)
        for n in range(n_min, n_max + 1):
            if L < n:
                continue
            grams.extend(t[i:i+n] for i in range(L - n + 1))
        return grams

    def _build_char_vocab(self, texts):
        counter = Counter()
        doc_counter = Counter()

        for txt in texts:
            grams = self._char_ngrams(txt)
            counter.update(grams)
            doc_counter.update(set(grams))

        min_freq = int(self.config.get("min_char_freq", 5))
        max_ratio = float(self.config.get("common_char_ratio", 1.0))  # default: don't filter by commonness
        max_vocab = self.config.get("max_char_vocab_size", None)
        total_length = len(texts)

        candidates = []
        for gram, freq in counter.items():
            if freq < min_freq:
                continue
            if total_length > 0 and (doc_counter[gram] / total_length) > max_ratio:
                continue
            candidates.append((gram, freq))

        if max_vocab is not None:
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[: int(max_vocab)]

        self.char_vocab = {gram: i for i, (gram, _) in enumerate(candidates)}
        self.char_vocab_size = len(self.char_vocab)

        if self.config.get("verbose_vocab", False):
            print(f"char vocab size: {self.char_vocab_size}")

    def compute_char_idf(self, texts):
        counts = np.zeros(self.char_vocab_size, dtype=float)
        total_length = len(texts)
        eps = self.config.get("epsilon", 1e-6)

        for txt in texts:
            grams = set(self._char_ngrams(txt))
            for g in grams:
                idx = self.char_vocab.get(g)
                if idx is not None:
                    counts[idx] += 1

        self.char_idf = np.log((total_length + 1) / (counts + 1 + eps)) + 1

    def _char_vector(self, text):
        vec = np.zeros(self.char_vocab_size, dtype=float)
        grams = self._char_ngrams(text)

        binary = self.config.get("binary_cts_char", False)
        use_tfidf = self.config.get("tfidf_char", False)

        for g in grams:
            idx = self.char_vocab.get(g)
            if idx is None:
                continue
            if binary:
                vec[idx] = 1
            else:
                vec[idx] += 1

        if use_tfidf and self.char_idf is not None:
            denom = vec.sum() + self.config.get("epsilon", 1e-6)
            tf = vec / denom
            vec = tf * self.char_idf
            norm = np.linalg.norm(vec) + self.config.get("epsilon", 1e-6)
            vec = vec / norm

        return vec


