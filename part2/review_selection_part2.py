#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import time
import spacy
import pandas as pd
import seaborn as sb
from tqdm import tqdm
import matplotlib.pyplot as plt
from spacy import displacy
from nltk.stem.snowball import SnowballStemmer
import math
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
import sklearn.preprocessing as preprocessing
from readability import Readability
import enchant
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import VotingClassifier
pd.options.mode.chained_assignment = None  # default='warn'
###Do Tokenization and stemming###
DATA_PATH = '/home/wangbo/桌面/sampled_data.csv'####Pls write your own path between''####

data = pd.read_csv(DATA_PATH)
data.rename(columns={'Unnamed: 0':'index'}, inplace=True)
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
    data.to_csv('/home/wangbo/桌面/token_stemming.csv')####Pls write your own path between''####
tokenizeReviews(data)
#####Find business ID with most review#####
data=pd.read_csv('/home/wangbo/桌面/token_stemming.csv')
data['business_id'].value_counts().max()
data['business_id'].value_counts().idxmax()
business_id1 = data['business_id']=='igHYkXZMLAc9UdV5VnR_AA'
business_id1
data1=data[business_id1]
#####Cut review with count<30 & >200####
text1 = (data1['stem_nums']<200)& (data1['stem_nums']>30)
data2=data1[text1]
#####Use Useful index as ylabel for further prediction#####
useful_nums = data2['useful'].value_counts()
i=0
data2['ylabel']=0
while i<=2417:
    if data2['useful'].iloc[i]>0:
        data2['ylabel'].iloc[i]=1.0
    else:
        data2['ylabel'].iloc[i]=0.0
    i+=1
ylabel_num=data2['ylabel'].value_counts()
ylabel_num
data2.to_csv('data2.csv')
nlp = spacy.load('en_core_web_sm')
####Get ADJ#####
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
#####Get Verb#####
def get_verb(review_list):
    review_text = review_list['text'].values
    total_len = len(review_list)
    print('len of review list', total_len)
    verb_list=[]
    pbar = tqdm(total=total_len)
    for doc in nlp.pipe(review_text, batch_size=400, n_process=16):
        verb=''
        for token in doc:
            if token.pos_ is 'VERB':
                verb+=token.lemma_                                                        
                verb+=' '
        verb=verb.strip(' ')
        verb_list.append(verb)
        pbar.update(1)
    pbar.close()
    verb_series=pd.Series(verb_list)
    review_list.insert(0,'verb',verb_series)
#####Get Noun#####
def get_noun(review_list):
    review_text = review_list['text'].values
    total_len = len(review_list)
    print('len of review list', total_len)
    noun_list=[]
    pbar = tqdm(total=total_len)
    for doc in nlp.pipe(review_text, batch_size=400, n_process=16):
        noun=''
        for token in doc:
            if token.pos_ is 'NOUN':
                noun+=token.lemma_                                                        
                noun+=' '
        noun=noun.strip(' ')
        noun_list.append(noun)
        pbar.update(1)
    pbar.close()
    noun_series=pd.Series(noun_list)
    review_list.insert(0,'noun',noun_series)
if __name__ == '__main__':
    review_list = pd.read_csv('data2.csv')
    get_adj(review_list)
    get_verb(review_list)
    get_noun(review_list)
#### ADD ADJ, VERB, NOUN Words COUNT#####
review_list['adj_count']=review_list.adj.apply(lambda x: len(str(x).split(' ')))
review_list['verb_count']=review_list.verb.apply(lambda x: len(str(x).split(' ')))
review_list['noun_count']=review_list.noun.apply(lambda x: len(str(x).split(' ')))
###################################################################################################
##Compute indivativeness and get top-10 most indicative adjectives for each rating star ##
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
    
###indicative adjective counts###
star1=['bad','cold','terrible','rude','horrible','ready','wrong','avaiable','rare','cooked']
star2=['cold','slow','disappointing','bad','many','horrible','raw','few','bland','well']
star3=['ok','good','nice','same','mediocre','bland','busy','average','cold','disappointed']
star4=['good','little','nice','only','small','great','large','first','quick','open']
star5=['amazing','favorite','delicious','perfect','great','wonderful','incredible','fantastic','top','beautiful']
review_list['adj_new']=review_list.adj.apply(lambda x: str(x).split(' '))
i=0
review_list['ind_adj']=0
while i<2418:
    if review_list['stars'].iloc[i]==1.0:
        for x in review_list['adj_new'].iloc[i]:
            if x in star1:
                review_list['ind_adj'].iloc[i]+=1
                
    elif review_list['stars'].iloc[i]==2.0:
        for x in review_list['adj_new'].iloc[i]:
            if x in star2:
                review_list['ind_adj'].iloc[i]+=1
    
    elif review_list['stars'].iloc[i]==3.0:
        for x in review_list['adj_new'].iloc[i]:
            if x in star3:
                review_list['ind_adj'].iloc[i]+=1     
                
    elif review_list['stars'].iloc[i]==4.0:
        for x in review_list['adj_new'].iloc[i]:
            if x in star4:
                review_list['ind_adj'].iloc[i]+=1
    
    elif review_list['stars'].iloc[i]==5.0:
        for x in review_list['adj_new'].iloc[i]:
            if x in star5:
                review_list['ind_adj'].iloc[i]+=1
                
    i+=1
    
####Flesh Kincaid Readability Test####
def FleschKincaidTest(text):
	score = 0.0
	if len(text) > 0:
		score = (0.39 * len(text.split()) / len(text.split('.')) ) + 11.8 * ( sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","y"] else 0,text))) / len(text.split())) - 15.59
		return score if score > 0 else 0
i=0
review_list['Readability']=0
while i<=2417:
    review_list['Readability'].iloc[i]=FleschKincaidTest(review_list['text'].iloc[i])
    i+=1
    
###wrong words counts in review###
d = enchant.Dict("en_US")
review_list['word']=review_list.text.apply(lambda x: str(x).split(' '))
i=0
review_list['notindic']=0
while i<=2417:
    for x in review_list['word'].iloc[i]:
        if len(x)==0:
            continue
        elif d.check(x)==False:
            review_list['notindic'].iloc[i]+=1
    i+=1

####Feature Engineering####
scaler = preprocessing.StandardScaler()
stem_nums_param = scaler.fit(review_list[['stem_nums']])
review_list['Scaled_stem_nums'] = scaler.fit_transform(review_list[['stem_nums']], stem_nums_param)
verb_count_param = scaler.fit(review_list[['verb_count']])
review_list['Scaled_verb_count'] = scaler.fit_transform(review_list[['verb_count']], verb_count_param)
noun_count_param = scaler.fit(review_list[['noun_count']])
review_list['Scaled_noun_count'] = scaler.fit_transform(review_list[['noun_count']], noun_count_param)
ind_adj_param = scaler.fit(review_list[['ind_adj']])
review_list['Scaled_ind_adj'] = scaler.fit_transform(review_list[['ind_adj']], ind_adj_param)
Readability_param = scaler.fit(review_list[['Readability']])
review_list['Scaled_Readability'] = scaler.fit_transform(review_list[['Readability']], Readability_param)
notindic_param = scaler.fit(review_list[['notindic']])
review_list['Scaled_notindic'] = scaler.fit_transform(review_list[['notindic']], notindic_param)
review_list.drop(['stem_nums','verb_count','noun_count','ind_adj','Readability','notindic'],axis=1, inplace=True)
review_list.drop(['review_id','user_id','business_id','stars','noun','verb','adj'],axis=1, inplace=True)
review_list.drop(['index','useful','funny','cool','text','date','token_nums'],axis=1, inplace=True)
review_list.drop(['adj_count','adj_new','word','Unnamed: 0', 'Unnamed: 0.1'],axis=1, inplace=True)

###Classifier Model Traning###
y = review_list['ylabel']
X = review_list.drop('ylabel',axis=1)
kfold=StratifiedKFold(n_splits=10)
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier(random_state=1))
classifiers.append(GradientBoostingClassifier(random_state=1))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(AdaBoostClassifier())
classifiers.append(GaussianNB())
for classifier in classifiers:
    cv_results = (cross_val_score(classifier,X,
                                  y,scoring='accuracy'
                                  ,cv=kfold,n_jobs=-1))
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % 
          ((cv_results.mean(), cv_results.std(), classifier)))
    
###hyper parameter adjustment###
#Logistic Regression
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(X,y)
print('modelgsLR score is：%.4f'%modelgsLR.best_score_)

###Final Model training###
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(random_state=2)
clf3 = GradientBoostingClassifier(random_state=1)
clf4 = SVC()
clf5 = AdaBoostClassifier()
clf6 = KNeighborsClassifier()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), 
                                    ('gnb', clf3), ('svc',clf4), ('Ada',clf5), 
                                    ('KNN', clf6)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, eclf], 
                      ['Logistic Regression', 'rf', 'GradientBoost', 'svc'
                       , 'Ada', 'KNN', 'Ensemble']):
    scores = cross_val_score(clf,X,y,
                             scoring='accuracy',cv=kfold,n_jobs=-1)
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % 
          (scores.mean(), scores.std(), label))
###Final Result####
clf1.fit(X,y)
preData=clf1.predict(X)
preData=preData.astype(int)
LRpreResultDf=pd.DataFrame()
LRpreResultDf['text']=data2['text']
LRpreResultDf['helpfulness']=preData
LRpreResultDf
i=0
LRpreResultDf['ranking']=0
while i <=2417:
    LRpreResultDf['ranking'].iloc[i]=review_list['Scaled_ind_adj'].iloc[i]-review_list['Scaled_Readability'].iloc[i]-review_list['Scaled_notindic'].iloc[i]
    i+=1
helpfulness1 = LRpreResultDf['helpfulness']==1
helpfulness1
LRpreResultDf1=LRpreResultDf[helpfulness1]
final_LRpreResultDf= LRpreResultDf1.sort_values(by=['ranking'], ascending=False)
pd.set_option('display.max_colwidth', None)
final_LRpreResultDf.head()


# In[ ]:




