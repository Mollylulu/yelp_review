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
from spacy import displacy
from nltk.stem.snowball import SnowballStemmer


DATA_PATH = './data/sampled_data.csv'

data = pd.read_csv(DATA_PATH)
data.rename(columns={'Unnamed: 0':'index'}, inplace=True)
print(data.head())
# print(data.columns)

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
    tokenNumPlot()


def tokenNumPlot():
    data = pd.read_csv('./data/token_stemming.csv')
    token_nums = data['token_nums'].value_counts()
    stem_nums = data['stem_nums'].value_counts()
    print('Most 5 Common Length of Review without Stemmed: {}'.format(token_nums.head(20).index))
    print('Most 5 Common Length of Review with Stemmed: {}'.format(stem_nums.head(20).index))

    def draw_dist():
        ax = sb.distplot(token_nums.index, kde=True, color="#D55E00", label="token num")
        sb.distplot(stem_nums.index, kde=True, color="#009E73", ax=ax, label='stemmed num')
        ax.legend()
        ax.set_xlim(0, 700)
        ax.set_xlabel('number of tokens/with stemming')
        # token_dataframe = token_nums.rename_axis('tokens').reset_index(name='counts')
        plt.legend()
        plt.savefig('./result/token_stemmed_dist.jpg')
        plt.show()

    def draw_token_count():
        plt.figure()
        ax1 = plt.subplot(211)
        plt.subplots_adjust(hspace=0.7)
        ax1.bar(token_nums.index, token_nums.values, facecolor="#F0E442", label='token num', width=5)
        plt.legend()
        ax1.set_xlabel('number of tokens')
        ax1.set_ylabel('counts of review')
        ax2 = plt.subplot(212)
        ax2.bar(stem_nums.index, stem_nums.values, facecolor="#009E73", label='stemmed num', width=5)
        ax2.set_xlabel('number of stemmed tokens')
        ax2.set_ylabel('counts of review')
        plt.legend()
        plt.savefig('./result/token_count.jpg')
        plt.show()

    draw_dist()
    # draw_token_count()


def starPlot():
    stars = data.value_counts('stars')
    assert isinstance(stars, pd.Series)
    stars.plot.pie(title='Stars among the data', label='', autopct='%1.1f%%',
                   colors=['#D55E00', '#E69F00', '#56B4E9', '#F0E442', '#009E73', ])
    plt.savefig('./result/stars_pct.jpg')
    plt.show()
    # print(stars)
    
    
if __name__ == '__main__':
    tokenNumPlot()
    # starPlot()
