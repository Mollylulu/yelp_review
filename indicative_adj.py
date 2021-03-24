# -*- coding: utf-8 -*-
# @Time    : 11/3/2021 下午 11:09
# @Author  : Wanlu Luo
# @FileName: indicative_adj.py


import spacy
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import nltk
from nltk import data
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt 
from pos_tagging import only_keep_words, get_clean_data


# nltk.download()
data.path.append(r"F:\packages\nltk")
nlp = spacy.load('en_core_web_sm')
DATA_PATH = './data/total_stars_text.csv'
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
# print(type(stop_words))


total_tokens = pickle.load(open('./data/total_pos_tokens_after_lemma.pkl', 'rb'))
star1_tokens = pickle.load(open('./data/star5_pos_tokens_after_lemma.pkl', 'rb'))

# star_total = len(star1_tokens)
# total = len(total_tokens)

star1_cfd_words = nltk.ConditionalFreqDist(star1_tokens)
total_cfd_words = nltk.ConditionalFreqDist(total_tokens)

total_adj_freq_dist = nltk.FreqDist()
adj_freq_dist = nltk.FreqDist()
noun_freq_dist = nltk.FreqDist()

adj_freq_dist.update(star1_cfd_words[wordnet.ADJ])
total_adj_freq_dist.update(total_cfd_words[wordnet.ADJ])

top100_adj_words = adj_freq_dist.most_common(100)

top100_adj_star = []
top100_adj_total = []
for word, freq in top100_adj_words:
    top100_adj_star.append(freq)
    top100_adj_total.append(total_adj_freq_dist[word])

top100_adj_star = np.array(top100_adj_star)
top100_adj_total = np.array(top100_adj_total)

star_total = np.sum(top100_adj_star)
total = np.sum(top100_adj_total)
p_w_r1 = top100_adj_star / star_total
p_w = top100_adj_total / total

relative_entropy = np.multiply(p_w_r1, np.log(p_w_r1/p_w))
indicative_dict = {}
for idx, (word, _) in enumerate(top100_adj_words):
    indicative_dict[word] = relative_entropy[idx]

indicative_dict = sorted(indicative_dict.items(), key=lambda kv: kv[1], reverse=True)
top10_star1_indicative_adj = dict(indicative_dict[:10])

indicative_adj = pd.DataFrame.from_dict(top10_star1_indicative_adj, orient='index', columns=['relative_entropy'])
indicative_adj = indicative_adj.reset_index().rename(columns={'index': 'word'})
print(indicative_adj.head(10))
ax = indicative_adj.plot(kind='bar', x='word', y='relative_entropy',
                         title='Top10 Indicative Adj in Star5 Reviews', figsize=(12, 11)
                         )

x_offset = 0.3
y_offset = 0.0005
for p in ax.patches:
    b = p.get_bbox()
    val = "{:.4f}".format(b.y1+b.y0)
    ax.annotate(val, ((b.x0 + b.x1)/2-x_offset, b.y1+y_offset))
plt.savefig('./result/star5_top10_indicative_ADJ.jpg')
plt.show()


def preprocess_text():
    stars_text = pd.read_csv(DATA_PATH, index_col=0)
    total_reviews = get_clean_data(stars_text['text'].values)
    star_1_reviews = stars_text[stars_text['stars'] == 1.0]['text']
    star_2_reviews = stars_text[stars_text['stars'] == 2.0]['text']
    star_3_reviews = stars_text[stars_text['stars'] == 3.0]['text']
    star_4_reviews = stars_text[stars_text['stars'] == 4.0]['text']
    star_5_reviews = stars_text[stars_text['stars'] == 5.0]['text']
    total_reviews = ' '.join(total_reviews)
    total_tokens, total_lens = tokenize_text(reviews=total_reviews)

    for i in range(5):
        star_reviews = only_keep_words(' '.join(eval('star_' + str(i + 1) + '_reviews.values')))
        star_tokens, _ = tokenize_text(reviews=star_reviews)
        star_tokens = get_lemmaed_pos_word_pairs(star_tokens)
        with open('./data/star' + str(i + 1) + '_pos_tokens_after_lemma.pkl', 'wb') as f:
            pickle.dump(star_tokens, f)


def tokenize_text(reviews):
    # assert isinstance(reviews, (list, np.ndarray))
    tokens = nltk.word_tokenize(reviews)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens, len(tokens)


def get_lemmaed_pos_word_pairs(tokens):
    tagged_words = nltk.pos_tag(tokens)
    for i, (word, pos) in enumerate(tagged_words):
        pos = get_wordnet_pos(tag=pos)
        word = lemmatizer.lemmatize(word, pos)
        tagged_words[i] = (pos, word)
    return tagged_words


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
