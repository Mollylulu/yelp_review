# -*- coding: utf-8 -*-
# @Time    : 24/2/2021 上午 11:59
# @Author  : Wanlu Luo
# @FileName: data_analysis.py


import time
import spacy
import pandas as pd
import seaborn as sb
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer


DATA_PATH = './data/sampled_data.csv'

data = pd.read_csv(DATA_PATH)
data.rename(columns={'Unnamed: 0':'index'}, inplace=True)
print(data.head())
print(data.columns)

nlp = spacy.load('en_core_web_sm')

def tokenizeReviews(data):
    token_nums = []
    stem_nums = []
    start_time = time.time()
    reviews = data['text'].values
    total_len = data.shape[0]
    stemmer = SnowballStemmer('english')
    print('Total reviews in this 200 business sub-dataset: {}'.format(total_len))
    pro_bar = tqdm(total=total_len)
    for doc in nlp.pipe(reviews, batch_size=200, n_process=8, disable=["parser", "ner", "textcat"]):
        _token = set()
        _stem = set()
        for token in doc:
            _token.add(token.text)
            _stem.add(stemmer.stem(token.text))
        token_nums.append(len(_token))
        stem_nums.append(len(_stem))
        pro_bar.update(1)
    pro_bar.close()
    data['token_nums'] = token_nums
    data['stem_nums'] = stem_nums
    assert isinstance(data, pd.DataFrame)
    data.to_csv('./data/token_stemming.csv')
    end_time = time.time()
    print('tokenizing time: {}'.format(end_time-start_time))
    tokenNumplot()


def tokenNumPlot():
    data = pd.read_csv('./data/token_stemming.csv')
    token_nums = data['token_nums'].value_counts()
    stem_nums = data['stem_nums'].value_counts()
    plt.figure()
    plt.subplot(211)
    p1 = sb.lineplot(x=token_nums.index, y=token_nums)
    p1.set_xlabel('number of tokens')
    p1.set_ylabel('counts of reviews')
    plt.subplot(212)
    p2 = sb.lineplot(x=stem_nums.index, y=stem_nums)
    p2.set_xlabel('number of tokens with stemming')
    p2.set_ylabel('counts of reviews')
    plt.savefig('./data/data_plot.jpg')
    plt.show()


def starPlot():
    stars = data.value_counts('stars')
    assert isinstance(stars, pd.Series)
    stars.plot.pie(title='Stars among the data', label='', autopct='%1.1f%%')
    plt.savefig('./data/stars_pct.jpg')
    plt.show()
    print(stars)
    
    
if __name__ == '__main__':
    # tokenNumPlot()
    starPlot()
