Exploration:
* check balance
* check format of text
* check for exact dupliactes or near duplicates in train/ val (avoid inflated f1)
* sentence length distribution (any long outliers)
* common words/phrases per class (profanity, slurs, negations, capitalization, punctuation)
* check for missingness 
* decide what can be filtered out (or slightly edited ex: @ -> <USER>) for better readability (e.g. punctuation, handles, emojis, etc.)

Preprocessing:
* Regex for text handling
* lowercasing
* define splits (train/validation)
* decide how we want to balance target (if need be)
* decide how we want to convert text into numeric data
* make all preprocessing steps toggleable so we can test different combinations/ easily rework after model evaluation

Feature extraction/ engineering:
* Bag of words
* TF-IDF
* N-grams (unigrams and bigrams)
* hand-engineered features (sentence length, %uppercase, count of punctuation, profanity indicators etc)

Baseline:
* Train/validate with baseline model(s) (TF-IDF + Logistic, bigrams + character n-grams Logistic, bag-of-words NN, SVM, etc.)
* Threshold tuning
* save: preprocessing version, model version/ weights, threshold, metrics for each iteration 
* According to competition rules, we cannot use public libraries or pretrained models so no sci-kit learn or models from huggingface or openai. Must keep in mind that we do have to build these models from scratch so we should not over burden ourselves. (But i think we can still use 

Validation strategy
* stratified splits (there is class imbalance)
* k-fold cross validation

Evaluation:
* We are being evaluated on by the F1 score so this will be our metric of importance (obviously we will use others but the submission is based off F1 score)
* Error analysis (manually inspect false positives, false positives, ambiguious cases, to refine preprocessing)

Extensions:
* extend to include a more complex model like a deep learning model (CNN/ MLP)
* fine tune our model of choice (we're going to do this regardless)

There are definitely some holes in this outline and I purposely left it pretty vague so that we can collab with our ideas easier so feel free to add to the outline or suggest ideas for certain tasks.

