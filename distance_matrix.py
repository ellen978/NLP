import pandas as pd
import numpy as np
import re 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import string
import collections
import nltk  
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import stopwords, wordnet
from nltk import PorterStemmer, WordNetLemmatizer  
import string
from decimal import *
import sys
from nltk.stem import PorterStemmer
from nltk import ne_chunk, pos_tag, word_tokenize
import collections
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)

#(1-1) DSN - claim level (67712);     
psel=pd.read_hdf('prop_cln.h5', 'text')
psel['TypeOfLossCode']=psel['TypeOfLossCode'].str.strip()  #for matching later, need this step on other side of dataset;
psel['wrdlen_c2']=psel['lossdespt_c2'].apply(lambda x: len(x.split()))  #67712; 

#(1-2) individual code check (major v.s. minor);
#major;
majors=['FR', 'LT', 'WN', 'HL', 'WT', 'FZ', 'I', 'CP', 'CO', 'TF', 'TO', 'TT', 'BR', 'VM', 'RB']  #remove RO; 
sub_majors=psel.loc[psel['TypeOfLossCode'].isin(majors), :]   
sub_majors['TypeOfLossCode'].value_counts() #16 categories, 47717/67712=70% of all claims; 

#minor;
minors=['VL', 'WR', 'HR', 'WS', 'SM', 'FM', 'SL', 'FW', 'MW'] 
sub_minors=psel.loc[psel['TypeOfLossCode'].isin(minors), :] 
sub_minors['TypeOfLossCode'].value_counts() #9 categories, 3367/67712=5% of all claims; 

#Per MM: remove 'JT', 'JB', 'IT', 'IB', 'JH', 'JL'; 

#major + minor 
cds=majors+minors
sub=psel.loc[psel['TypeOfLossCode'].isin(cds), :]   #25 groups, 75%;   
others=psel.loc[~psel['TypeOfLossCode'].isin(cds), :]   #111 groups, 25%; 

# (1-3) distance matrix check (tpcd level for sub); 
mx1=sub.loc[:, ['TypeOfLossCode', 'lossdespt_c2']].groupby('TypeOfLossCode').agg(' '.join)  
mx1['wrdlen_c2']=mx1['lossdespt_c2'].apply(lambda x: len(x.split()))  
mx1['TypeOfLossCode']=mx1.index   
labs=mx1['TypeOfLossCode']

tfidfv=TfidfVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english', strip_accents='unicode', norm='l2')
mx2=tfidfv.fit_transform(mx1['lossdespt_c2']).todense()
col = [i for i in tfidfv.get_feature_names()] 
print(mx2.shape)

mx2_df=pd.DataFrame(mx2, columns=col)
cs_matrix=cosine_similarity(mx2_df) 
print(cs_matrix.shape)   
cs_f1=pd.DataFrame(cs_matrix, index=labs, columns=labs) 

# (1-4) Check of top words per TypeOfLossCode;     
def dfdic(text):
    dic={}
    for i in text:
        for j in set(i.split()):
            if j in dic:
                dic[j]+=1
            else:
                dic[j]=1 
    return dic

rank=20  #top 10 words; 
res={}
wrds=set()  #all wrds, for loopings;  
for c in cds:
    dic=dfdic(sub.loc[sub['TypeOfLossCode']==c, 'lossdespt_c2'])
    line=sorted(dic.values(), reverse=True)[rank:(rank+1)][0]
    keywrd=[k for k, v in dic.items() if v>line]
    res[c]=keywrd
wrd_per_group=pd.DataFrame(res.items(), columns=['group', 'topwrds']) 
wrd_per_group
