# -*- coding: utf-8 -*-
# @Time    : 11/3/2021 上午 11:27
# @Author  : Wanlu Luo
# @FileName: pos_tagging.py


import spacy
import random
from tqdm import tqdm
from spacy import displacy
import pandas as pd
import numpy as np
from pathlib import Path
import nltk

nlp = spacy.load('en_core_web_sm')

def pos_tagging():
    DATA_PATH = './data/sampled_pos.csv'
    np.random.seed(1763)

    review_samples = pd.read_csv(DATA_PATH)['text'].tolist()
    # rand_idx = np.random.randint(0, data.shape[0], size=(3,))
    # print(rand_idx)
    # print(data.columns)
    # review_samples = data.iloc[rand_idx]['text'].tolist()
    # print(review_samples)
    reviews = list(nlp.pipe(review_samples, disable=['ner', 'textcat']))
    # # for review in reviews:
    sentence_spans = list(reviews[0].sents)
    for i in sentence_spans:
        print(i)

    # # displacy.serve(sentence_spans, style='dep')
    # options = {"compact": True, "color": "blue"}
    # pos_tagging_img = displacy.render(sentence_spans, style="dep", page="true", options=options)
    # pos_tagging_path = Path('./data/pos_tagging.svg')
    # pos_tagging_path.open("w", encoding="utf-8").write(pos_tagging_img)

def count_reviews():
    from collections import Counter
    # all_stopwords = spacy.Defaults.stop_words
    data = pd.read_csv('./data/sampled_data.csv')
    reviews = data['text'].values
    print('reviews data type', type(reviews))
    total_len = data.shape[0]
    print('Total reviews in this 200 business sub-dataset: {}'.format(total_len))
    # pro_bar = tqdm(total=total_len)
    total_tokens = []
    reviews_processed = get_clean_data(reviews)
    docs = nlp.pipe(reviews, batch_size=200, n_process=8, disable=["parser", "ner", "textcat"])
    noun_tokens = []
    adj_tokens = []
    for doc in tqdm(docs):
        for token in doc:
            if token.pos_ in('NOUN', 'ADJ'):
                if token.is_alpha and len(token.text)>2 and not token.is_stop:
                    eval(token.pos_.lower()+'_tokens'+'.append(token.text.lower())')
        # pro_bar.update(1)

    noun_word_freq = Counter(noun_tokens)
    adj_word_freq = Counter(adj_tokens)
    noun_common_words = noun_word_freq.most_common(10)
    adj_common_words = adj_word_freq.most_common(10)
    print('Most 10 NOUN common words in this sampled dataset: ')
    print(noun_common_words)
    print('Most 10 ADJ common words in this sampled dataset: ')
    print(adj_common_words)

    # def write_data(data_type):
    #     file_name = 'top10_'+data_type+'.txt'
    #     with open(file_name, 'a') as f:
    #         for word in eval(data_type+'_common_words'):
    #             f.write(word)
    #             f.write('\n')
    #
    # write_data(data_type='noun')
    # write_data(data_type='adj')


def only_keep_words(review_text):
    import re
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    review_text = re.sub("\s+", ' ', review_text)
    review_text = ' '.join(re.findall('\w{3,}', review_text))
    return review_text


def get_clean_data(data):
    data = data.tolist()
    processed_data = []
    for text in data:
        processed_data.append(only_keep_words(text))
    return processed_data


if __name__ == '__main__':
    count_reviews()
    # pos_tagging()