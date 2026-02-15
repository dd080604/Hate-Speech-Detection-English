
FEATURE_CONFIG = {
    "feature_set_name": "word12",

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

    "use_word_features": True,
    "tfidf": True,
    "bigrams": True,
    "binary_cts": False,

    #vocab settings
    "min_word_freq": 3,
    "common_word_ratio": 0.90,
    "max_vocab_size": None,

    ## Char N-Gram Features
    "char_n_grams":False,
    "char_n_min": 3,
    "char_n_max": 5,
    "min_char_freq": 5,

    ### Numeric Features

    # "use_engineered_features":  True,
    #
    # "use_length_features":  True,
    # "use_uppercase_ratio":  True,
    # "use_punctuation_counts":  True,
    # "use_repeated_char_feature": True,
    # "use_profanity_features": True,
    # "use_negation_feature": True,
    # "use_special_token_flags": True,

    ## Normalization Settings
    "normalize_features": True,
    "epsilon": 1e-6,

    # DEBUG
    "verbose_vocab": True,
    "verbose_features": True,
}