

# yelp_review

this is a assignment from TEXT MANAGEMENT class, which handles the `yelp review` analysis.

# I. Requirement

```
# spacy
pip install -U spacy

# language model
python -m spacy download en_core_web_sm

# pandas seaborn tqdm pickle
pip install pandas seaborn tqdm pickle

# nltk
# https://www.nltk.org/install.html
pip install nltk
ntlk.download()

# torch
## we recommend you with the conda environment
conda install -c pytorch pytorch

# transformers
conda install -c conda-forge transformers

# lime
conda install -c conda-forge lime

# scikit learn
pip install -U scikit-learn
https://scikit-learn.org/stable/install.html

# redability
pip install py-readability-metrics
https://pypi.org/project/py-readability-metrics/

# enchant
pip install pyenchant
https://pypi.org/project/pyenchant/

```

# II. Tasks

## Part1

- [x] **Subset Selection**
  refers to the `randrom_collect.py`
  outcome is `./data/sampled_data.csv`

- [x] **Data Analysis**

  refers to the `data_analysis.py`. It plots the figures about the length distribution among all the data.

- [x] **POS Tagging**

  refers to the `pos_tagging.py`. It tags the reviews.

- [x] **Indicative Adjectives**
  refers to the `get_top10_adj.py`

  

## Part2

- [x] **Top 5 Review Selection Model**
  
  refers to the `review_selection_part2.py`

## Part3

- [x] **Sentiment Analysis**

  refers to `sentiment_q3.ipynb`

