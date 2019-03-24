
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
  
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 500)


all=pd.read_hdf('text.h5', 'text')   
sel=all.loc[all['ClaimType']=='Prop_Claim', ['LossDescription', 'TypeOfLossCode', 'TypeofLossCodeDescription']] 
sel.drop_duplicates(inplace=True)  

#********Step 1; 
sel['lossdespt_c'] = sel['LossDescription'].fillna('') 
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(re.sub(r'[^\w]',' ',x).split())) #rm punc;


#*******Step 2: chunking; 
def nmchunk(ts):
    res={}
    person=set()
    gpe=set()
    for text in ts:
        nec=ne_chunk(pos_tag(text.split()))    #word_tokenize(text)
        for chunk in nec:
              if hasattr(chunk, 'label'):
                    if chunk.label()=='PERSON':
                        person.add(chunk[0][0])
                    elif chunk.label()=='GPE':
                        gpe.add(chunk[0][0])
        res['person']=person
        res['gpe']=gpe
    return res

# chunkdic=nmchunk(sel['lossdespt_c'].fillna(''))
# prndic=chunkdic['person']   #4853;   
# gpedic=chunkdic['gpe']   #2471;   

# with open('allnmchunks.pickle', 'wb') as pk:
#     pickle.dump(chunkdic, pk)

with open('allnmchunks.pickle', 'rb') as pkr:
    chunkdic = pickle.load(pkr)
    prndic=chunkdic['person']  
    gpedic=chunkdic['gpe']


#clean prndic; 
from nltk.corpus import wordnet
nm=set()
eng=set()
for i in list(prndic):
    if not wordnet.synsets(i):
        nm.add(i)
    else:
        eng.add(i)
    #2262 nms v.s. 2591 eng; 


from nltk.corpus import words
nm2=set()
eng2=set()
t=set(words.words())
for i in list(prndic):
    if i in t:
        eng2.add(i)
    else: 
        nm2.add(i)
    #3996 nm2 v.s. 857 eng2;

nms=nm-nm2 #yes: 376
nms_add={'Roberta', 'Roberto', 'Robert', 'Roberts', 'Robertson'}
nms_all=nms | nms_add  #379  
    #to extend nms_all: enchant or PyDictionary for 3rd & 4th check); 
    
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(c for c in x.split() if c not in nms_all))


#*******Step 3  
a=set(stopwords.words('english'))
sel['lossdespt_c']=sel['lossdespt_c'].str.lower()
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(c for c in x.split() if not c.isdigit())) #rm num; 
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: [c for c in x.split() if c not in a])
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: [c for c in x if not c.isdigit()])
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(x))

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
    
def normalize(text):  
    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    lemm_words = [nltk.WordNetLemmatizer().lemmatize(w[0], get_wordnet_pos(w[1])) for w in word_pos]  
    return [x.lower() for x in lemm_words]

sel['lossdespt_c']=sel['lossdespt_c'].apply(normalize)
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(x))


#*******Step 4 (based on lemma, more apprehensible)
def dfdic(text):  
    dic={}
    for i in text: 
        for j in set(i.split()):
            if j in dic:
                dic[j]+=1
            else:
                dic[j]=1  
    return dic
dic_c=dfdic(sel['lossdespt_c'])  #for rm1 only (later use need to update);  

rm1={k for k, v in dic_c.items() if re.match(r'cl.*m.*t.*', k) and v!=1 and k not in {'climt', 'clements', 'climate'}} #37
rm2={'isrd', 'insdureds', 'insures', 'insurer', 'insuree', 'insured', 'insred', 'insurance', 'insuranc', 
     'insrued', 'insurnace', 'indsured', 'inuserd', 'insureds', 'iinsured', 'insuerd', 'insurred', 'insrds', 
     'insruance', 'isnured', 'inusured', 'insr', 'insureance', 'instrusion', 'insur', 'inusred', 'isured', 
     'insurds', 'insrd', 'insusred', 'insuraed', 'insure', 'insurd', 'insurors', 'isnrd'}
rm3={'alledge', 'allegedly', 'alleages', 'alleged', 'alleges', 'alledgely', 'allegery', 'allges', 'allesges', 
    'allesged', 'alledgedly', 'alleg', 'alledges', 'alledged', 'alleging', 'allegeding', 'allegation', 'aleging',
    'allegs', 'allege', 'allegely', 'aleges', 'allgede', 'allgeing', 'allegly', 'allegedes', 'alegges', 
     'allegdly', 'allegedy'}
rm4={k for k, v in dic_c.items() if len(k)==1}
rm5={'insd', 'cmt', 'insd', 'claim', 'damage', 'cause'}
r2=rm1 | rm2 | rm3 | rm4 | rm5  #101

sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(c for c in x.split() if c not in r2))   #23,969


#*******Step 5: further ptstm, rm freq==1 (more precise); 
dic_c2=dfdic(sel['lossdespt_c'])  #updated: 23,969;  
porter=PorterStemmer()
dic_p=collections.Counter()  
lp_map={}

for k, v in dic_c2.items():  
    dic_p[porter.stem(k)]+=v
    lp_map[k]=porter.stem(k)
    
rm7={k for k, v in dic_p.items() if v==1}  #r3: 9536 (14,433 to stay)

r3={k for k, v in lp_map.items() if v in rm7}  #use for lemma rm; 
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(c for c in x.split() if c not in r3)) #14,433;


#*******Step 6: use ptstm to further normalize lemma by lemma_c (dependt on results from CS5)  
lp_map_2={k:v for k, v in lp_map.items() if v not in rm7} #cleaned 14,433 lemma & ptstm xwalk; 
lp=pd.DataFrame(lp_map_2.items(), columns=['lemma', 'ptstm'])
lp.sort_values(by='ptstm', inplace=True)  

lp['len_lemma']=lp['lemma'].apply(len)
lp_u=lp.sort_values(by=['ptstm', 'len_lemma'], ascending=True).drop_duplicates(['ptstm'], keep='first')  #11,006, use to replace lemmatized words; 

lp_u.drop('len_lemma', axis=1, inplace=True)
lp_u.rename(columns={'lemma': 'lemma_c'}, inplace=True)  #updated xwalk; 
lp.drop('len_lemma', axis=1, inplace=True)  #original xwalk; 

## df, dic; 
lp_xwalk=lp.merge(lp_u, how='outer', on='ptstm')  #lemma1 & lemma_c(ptstm) mapping: 14,433 v.s. 11,006; 
lpxwalk_dic = dict(zip(lp_xwalk['lemma'], lp_xwalk['lemma_c']))


#Both lossdespt_c (14,430) and lossdespt_c2 (11,006) kept, can be used interchangably; 
sel['lossdespt_c2']=sel['lossdespt_c'].apply(lambda x: ' '.join(lpxwalk_dic[c] for c in x.split()))


#*******Step 7: removal based off iterative cleasing;   
#lpxwalk_dic (map for normalized lemma(c)-stem(c2) mapping);    
  
####from _c2,  
r4_v={'dmg', 'unit', 'ownership', 'call', 'due', 'bldg', 'statu', 'file', 'review', 'cc', 'get', 
      'date', 'messag', 'take', 'release', 'back', 'possibl', 'go', 'send', 'type', 'text', 'come',
      'report', 'subject', 'receiv', 'temporary', 'user', 'read', 'approv', 'build', 'request', 
     'letter', 'see', 'note', 'state', 'dr'}  #for _c2

r4_k={k for k, v in lpxwalk_dic.items() if v in r4_v}  #for _c;    


##lossdespt_c (14,334) and lossdespt_c2 (10,970)      
sel['lossdespt_c']=sel['lossdespt_c'].apply(lambda x: ' '.join(c for c in x.split() if c not in r4_k))
sel['lossdespt_c2']=sel['lossdespt_c2'].apply(lambda x: ' '.join(c for c in x.split() if c not in r4_v))


sel.to_hdf('prop_clm.h5', key='text', mode='w')  ###RUN AS LAST STEP!!!       
