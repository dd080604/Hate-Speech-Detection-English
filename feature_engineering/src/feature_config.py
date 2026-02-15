BASE_CONFIG = {
    "val_size": 0.2,
    "split_seed": 42,
    "numeric_feature_cols": [
        "char_len",
        "word_len",
        "shout_ratio",
        "punct_count",
        "url_count",
        "handle_count",
        "hashtag_count",
        "emoji_count",
        "profanity_hits",
        "slur_hits",
        "negation_hits",
    ],
    "normalize_features": True,
    "epsilon": 1e-6,

    "verbose_vocab": True,
    "verbose_features": True,

    # word controls
    "min_word_freq": 3,
    "common_word_ratio": 0.90,
    "max_vocab_size": None,

    # char controls
    "char_n_min": 3,
    "char_n_max": 5,
    "min_char_freq": 5,
    "common_char_ratio": 1.0,
    "max_char_vocab_size": None,
}

FEATURE_SETS = {
    "word12": {
        "use_word_features": True,
        "tfidf_word": True,
        "bigrams": True,
        "binary_cts_word": False,

        "use_char_features": False,
        "tfidf_char": False,
    },
    "char35": {
        "use_word_features": False,

        "use_char_features": True,
        "tfidf_char": True,
        "binary_cts_char": False,
    },
    "word12_char35": {
        "use_word_features": True,
        "tfidf_word": True,
        "bigrams": True,

        "use_char_features": True,
        "tfidf_char": True,
    },
}

def get_feature_config(name: str):
    cfg = dict(BASE_CONFIG)
    cfg["feature_set_name"] = name
    cfg.update(FEATURE_SETS[name])
    return cfg

# Pick which one you want to run:
FEATURE_CONFIG = get_feature_config("word12")