Exploration:
* check balance
* check format of text
* decide what can be filtered out for better readability (e.g. punctuation, handles, emojis, etc.)

Preprocessing:
* Regex for text handling
* define splits (train/validation)
* decide how we want to balance target (if need be)
* decide how we want to convert text into numeric data

Baseline:
* Train/validate with baseline model(s) (Logistic, SVM, etc.)
* According to competition rules, we cannot use public libraries or pretrained models so no sci-kit learn or models from huggingface or openai. Must keep in mind that we do have to build these models from scratch so we should not over burden ourselves.

Evaluation:
* We are being evaluated on by the F1 score so this will be our metric of importance (obviously we will use others but the submission is based off F1 score)

Extensions:
* extend to include a more complex model like a deep learning model
* fine tune our model of choice (we're going to do this regardless)

There are definitely some holes in this outline and I purposely left it pretty vague so that we can collab with our ideas easier so feel free to add to the outline or suggest ideas for certain tasks.

