import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import spacy

##################################################################                                               
####### Step 1: select all the adjectives from each review #######
##################################################################
nlp = spacy.load('en_core_web_sm')

def get_adj(review_list):
    review_text = review_list['text'].values
    total_len = len(review_list)
    print('len of review list', total_len)
    adj_list=[]
    pbar = tqdm(total=total_len)
    for doc in nlp.pipe(review_text, batch_size=400, n_process=16):
        adj=''
        for token in doc:
            if(token.pos_ is 'ADJ' and (token.dep_ is 'amod' or token.dep_ is 'acomp')): #amod: adjectival modifier
                adj+=token.lemma_                                                        #acomp: adjectival complement
                adj+=' '
        adj=adj.strip(' ')
        adj_list.append(adj)
        pbar.update(1)
    pbar.close()
    adj_series=pd.Series(adj_list)
    review_list.insert(0,'adj',adj_series)
    review_list.to_csv('adj_output.csv')

if __name__ == '__main__':
    review_list = pd.read_csv('sampled_data.csv')
    get_adj(review_list)


###################################################################################################
## Step 2: compute indivativeness and get top-10 most indicative adjectives for each rating star ##
###################################################################################################

# the probability of observing word in all reviews
def compute_prob_in_all(adj, word): 
    word_num = 0
    for i in adj:
        if(word in i):
            word_num += 1
    prob_in_all = word_num / len(adj)
    return prob_in_all

# the probability of observing word in all reviews with rating star
def compute_prob_in_stars(adj, stars, current_star, word): 
    word_num = 0
    stars_num = 0
    for i in range(len(adj)):
        if(stars[i] == current_star):
            stars_num += 1
            if(word in adj[i]):
                word_num += 1
    prob_in_stars = word_num / stars_num
    return prob_in_stars

reviews = pd.read_csv('adj_output.csv')
adj_reviews = reviews['adj'].values

adj = [] # each line represents adjectives extracted from each review
for i in adj_reviews:
    if(pd.isnull(i)):
        adj.append([])
    else:
        adj.append(i.split(' '))

all_adj_word = set()  # all non-repeated adjectives that appear in reviews
for i in adj:
    for j in i:
        all_adj_word.add(j)

stars = reviews['stars'].values
all_stars=[1.0,2.0,3.0,4.0,5.0]

# compute indivativeness and get top-10 results
for current_star in all_stars:
    word_IA = dict() # key:word value:IA(indicative adjectives)
    for word in all_adj_word:
        prob_in_stars = compute_prob_in_stars(adj, stars, current_star, word)
        prob_in_all = compute_prob_in_all(adj, word)
        if(prob_in_stars == 0):
            prob_indicative = 0
        else:
            prob_indicative = prob_in_stars * math.log(prob_in_stars / prob_in_all)
        word_IA[word] = prob_indicative
    result = sorted(word_IA.items(), key=lambda d:d[1], reverse=True)
    print('top-10 most indicative adjectives for star', current_star, ':')
    print(result[0:10],'\n')

