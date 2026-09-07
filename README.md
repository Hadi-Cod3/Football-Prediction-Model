# Football Match Outcome Prediction

A baseline machine learning model that predicts whether an international football match ends in a home win, away win, or draw, using historical match data from 1872 to 2017.

This was built as a first end-to-end ML project, covering data loading, cleaning, feature encoding, model training, and evaluation, with the goal of demonstrating a real, working process rather than just a final result.

## Dataset

* **Source:** [International Football Results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle)
* **Size:** \~49,547 matches
* **Columns used:** `date`, `home\_team`, `away\_team`, `home\_score`, `away\_score`, `neutral`

No missing values were found in any column during initial checks.

## Approach

### 1\. Target variable

Since the dataset only provides scores, a `result` column was created by comparing `home\_score` and `away\_score` to label each match as `home\_win`, `away\_win`, or `draw`.

Class distribution:

|Outcome|Count|Share|
|-|-|-|
|Home win|24,276|\~49%|
|Away win|14,010|\~28%|
|Draw|11,261|\~23%|

This confirmed a real home-field advantage in the data before any modeling began.

### 2\. Attempt 1: Label encoding (failed)

Team names were first converted to numbers using `LabelEncoder`. The resulting model scored 49% accuracy, but a closer look at the classification report showed it was only ever predicting `home\_win`, with precision and recall at 0 for both other classes.

**Root cause:** `LabelEncoder` assigns arbitrary integers to each team (e.g. Scotland = 252, England = 89), which introduces a false numeric ordering. The model was treating team identity as if it were ordinal data, when it isn't.

### 3\. Attempt 2: One-hot encoding (fixed)

Switched to one-hot encoding, giving each team its own independent true/false column instead of a single arbitrary number. This removed the false ranking and let the model learn from team identity properly.

## Results

|Metric|Attempt 1 (Label Encoding)|Attempt 2 (One-Hot Encoding)|
|-|-|-|
|Accuracy|49%|58%|
|Home win recall|100% (degenerate)|85%|
|Away win recall|0%|55%|
|Draw recall|0%|4%|

The final model (Attempt 2) meaningfully outperforms the naive baseline and shows real signal: it distinguishes between home and away wins rather than defaulting to the majority class.

**Known limitation:** draws remain very hard to predict (4% recall). This is expected, since draws don't correlate as cleanly with team strength as wins do; two closely matched teams are close to a coin flip regardless of historical form.

## Tech stack

Python, pandas, scikit-learn (`LogisticRegression`, `train\_test\_split`, `classification\_report`)

## How to run

bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Football\_Prediction\_model.ipynb

