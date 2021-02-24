# -*- coding: utf-8 -*-
# @Time    : 24/2/2021 上午 11:59
# @Author  : Wanlu Luo
# @FileName: load_dataset.py


import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
import time
import seaborn as sb


DATA_PATH = './data/sampled_data.csv'

data = pd.read_csv(DATA_PATH)
data.rename(columns={'Unnamed: 0':'index'}, inplace=True)
print(data.head())
print(data.columns)

nlp = spacy.load('en_core_web_sm')

def tokenizeReviews(data):
    token_nums = []
    lemma_nums = []
    start_time = time.time()
    reviews = data['text'].values
    total_len = data.shape[0]
    print('Total reviews in this 200 business sub-dataset: {}'.format(total_len))
    pro_bar = tqdm(total=total_len)
    for doc in nlp.pipe(reviews, batch_size=100, n_process=8, disable=["parser", "ner", "textcat"]):
        _token = set()
        _lemma = set()
        for token in doc:
            _token.add(token.text)
            _lemma.add(token.lemma_)
        token_nums.append(len(_token))
        lemma_nums.append(len(_lemma))
        pro_bar.update(1)
    pro_bar.close()
    data['token_nums'] = token_nums
    data['lemma_nums'] = lemma_nums
    assert isinstance(data, pd.DataFrame)
    data.to_csv('./data/token_lemma.csv')
    end_time = time.time()
    print('tokenizing time: {}'.format(end_time-start_time))
    plot()


def plot():
    data = pd.read_csv('./data/token_lemma.csv')
    token_nums = data['token_nums'].value_counts()
    lemma_nums = data['token_nums'].value_counts()
    plt.figure()
    p1 = sb.lineplot(x=token_nums.index, y=token_nums)
    p1.set_xlabel('number of tokens')
    p1.set_ylabel('counts of reviews')
    plt.figure()
    p2 = sb.lineplot(x=lemma_nums.index, y=lemma_nums)
    p2.set_xlabel('number of tokens with stemming')
    p2.set_ylabel('counts of reviews')
    plt.savefig('./data/data_plot.jpg')
    plt.show()
    
    
if __name__ == '__main__':
    tokenizeReviews(data=data)