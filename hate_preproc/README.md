## 1. Clone the repository, Set directory, Install:

```bash
!git clone https://github.com/dd080604/Hate-Speech-Detection-English.git
%cd Hate-Speech-Detection-English/hate_preproc
%pip install -e .
```
* You only need clone the repo once.
* For the second line, your path may look a little different but make sure it includes at least the two files above.

## 2. Example Usage

```python
import pandas as pd
from hate_preproc import PreprocessConfig, preprocess_dataframe, add_signal_columns

df = pd.read_csv("hate_speech_train.csv")
cfg = PreprocessConfig()
df_clean = preprocess_dataframe(df, cfg)
```
This outputs a dataframe with the original data plus the column with clean text:
<img width="758" height="183" alt="image" src="https://github.com/user-attachments/assets/52a65d48-96fc-422c-8c8c-123bb7e2f540" />

## 3. PreprocessConfig

Each of these arguments can be toggled in the creation of the PreprocessConfig instance:
```python
cfg = PreprocessConfig(
    lowercase=True,
    strip_whitespace=True,
    remove_urls=True,
    remove_handles=True,
    normalize_hashtags=True,
    remove_emojis=False,
    remove_numbers=False,
    remove_punctuation=False,
    collapse_repeated_chars=True,
    normalize_quotes=True
)
# These are the default settings when nothing is toggled in cfg.
```

## 4. Feature Signals

The `add_signal_columns` function takes the uncleaned data and extracts certain features that may or may not aid our models in prediction. 

```python
df_signals = add_signal_columns(df, cfg)
df_signals.head()
```
<img width="1355" height="347" alt="output-onlinepngtools" src="https://github.com/user-attachments/assets/c5086266-35c3-4f43-a568-d010e8a326c8" />



