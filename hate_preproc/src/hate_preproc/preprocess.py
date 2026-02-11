# Hate Speech Text Pipeline

import re
import string
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Configuration

@dataclass
class PreprocessConfig:
    # Column names
    text_col: str = "text"
    label_col: str = "label"
    id_col: str = "id"

    # Cleaning toggles
    lowercase: bool = True
    strip_whitespace: bool = True
    remove_urls: bool = True
    remove_handles: bool = True          # "@"username
    normalize_hashtags: bool = True      # keep word but drop "#"
    remove_emojis: bool = False          # set True if emojis are mostly noise
    remove_numbers: bool = False         # set True if numbers are unhelpful
    remove_punctuation: bool = False     # often False for hate speech, punctuation can be informative
    collapse_repeated_chars: bool = True # "soooo" -> "soo"
    normalize_quotes: bool = True
    random_state: int = 42

CFG = PreprocessConfig()


# Feature signals (before cleaning)

# (placeholder lists)
PROFANITY = {"fuck", "shit", "bitch", "asshole", "dick", "piss", "cunt", "bastard"}
SLURS = {
    # Idk about this one yet
}

NEGATIONS = {"not", "no", "never", "none", "n't", "cannot", "can't", "don't",
             "won't", "isn't", "aren't", "wasn't", "weren't", "cant", "dont",
             "wont", "isnt", "arent", "wasnt", "werent"}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
HANDLE_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "["                  # basic emoji blocks
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE
)

def shout_ratio(s: str) -> float:
    if not s:
        return 0.0
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    uppers = sum(ch.isupper() for ch in letters)
    return uppers / len(letters)

def count_punct(s: str) -> int:
    return sum(ch in string.punctuation for ch in s) if s else 0

def vocab_hits(s: str, lex: set) -> int:
    if not s:
        return 0
    toks = re.findall(r"[A-Za-z']+", s.lower())
    return sum(t in lex for t in toks)

def negation_hits(s: str) -> int:
    if not s:
        return 0
    toks = re.findall(r"[A-Za-z']+", s.lower())
    return sum(t in NEGATIONS for t in toks)

def count_urls(s: str) -> int:
    return len(URL_RE.findall(s)) if s else 0

def count_handles(s: str) -> int:
    return len(HANDLE_RE.findall(s)) if s else 0

def count_hashtags(s: str) -> int:
    return len(HASHTAG_RE.findall(s)) if s else 0

def count_emojis(s: str) -> int:
    return len(EMOJI_RE.findall(s)) if s else 0

def add_signal_columns(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    x = df.copy()
    s = x[cfg.text_col].fillna("").astype(str)

    x["char_len"] = s.str.len()
    x["word_len"] = s.apply(lambda t: len(re.findall(r"\b\w+\b", t)))
    x["shout_ratio"] = s.apply(shout_ratio)
    x["punct_count"] = s.apply(count_punct)
    x["url_count"] = s.apply(count_urls)
    x["handle_count"] = s.apply(count_handles)
    x["hashtag_count"] = s.apply(count_hashtags)
    x["emoji_count"] = s.apply(count_emojis)

    x["profanity_hits"] = s.apply(lambda t: vocab_hits(t, PROFANITY))
    x["slur_hits"] = s.apply(lambda t: vocab_hits(t, SLURS))
    x["negation_hits"] = s.apply(negation_hits)
    return x


# Cleaning

def normalize_text(text: str, cfg: PreprocessConfig) -> str:
    if text is None:
        return ""

    t = str(text)

    if cfg.normalize_quotes:
        t = t.replace("’", "'").replace("“", '"').replace("”", '"')

    if cfg.remove_urls:
        t = URL_RE.sub(" ", t)

    if cfg.remove_handles:
        t = HANDLE_RE.sub(" ", t)

    if cfg.normalize_hashtags:
        # "#word" -> "word"
        t = HASHTAG_RE.sub(r"\1", t)

    if cfg.remove_emojis:
        t = EMOJI_RE.sub(" ", t)

    if cfg.collapse_repeated_chars:
        t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)

    if cfg.lowercase:
        t = t.lower()

    if cfg.remove_numbers:
        t = re.sub(r"\d+", " ", t)

    if cfg.remove_punctuation:
        if cfg.keep_apostrophes:
            punct = string.punctuation.replace("'", "")
        else:
            punct = string.punctuation
        t = t.translate(str.maketrans({ch: " " for ch in punct}))

    if cfg.strip_whitespace:
        t = re.sub(r"\s+", " ", t).strip()

    return t


def preprocess_dataframe(df: pd.DataFrame, cfg: PreprocessConfig, out_col: str = "text_clean") -> pd.DataFrame:
    x = df.copy()
    x[out_col] = x[cfg.text_col].apply(lambda s: normalize_text(s, cfg))
    return x

