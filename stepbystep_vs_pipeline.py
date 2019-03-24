data=pd.read_csv('claims.csv')
data=data.fillna('')

#Approach 1: step by step
data=pd.read_csv('claims.csv')
data['despt_c'] = data['Description'].fillna('')
data['despt_c']=data['despt_c'].str.lower()
data['despt_c']=data['despt_c'].str.split()
data['despt_c']=data['despt_c'].apply(lambda x: [re.sub('[^a-zA-Z]*', '', i) for i in x])
data['despt_c']=data['despt_c'].apply(lambda x: [c for c in x if not c.isdigit()])

sw=set(stopwords.words('english'))-{'no', 'nor', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 
                                 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
                                 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't",
                                  'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
                                 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
data['despt_c']=data['despt_c'].apply(lambda x: [c for c in x if c not in sw])
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

data['despt_c']=data['despt_c'].apply(lambda x: ' '.join(x))
data['despt_c']=data['despt_c'].apply(normalize)
data['despt_c']=data['despt_c'].apply(lambda x: ' '.join(x))




#Approach 2: pipelines 
def processing(text):
    def stopword(text):  
        negwrd={'no', 'nor', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
        'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', 
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
        "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}  
        allstopwrd=set(stopwords.words('english')) 
        stopwrd=allstopwrd - negwrd 
        nonstop=[c for c in text.lower().split() if c not in stopwrd]
        return ' '.join(nonstop)
    
    def remove(text):
        wl=' '.join([re.sub('[^a-zA-Z]*', '', c) for c in text.lower().split() if (not c.isdigit())])
        return wl
    
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
        word_pos = nltk.pos_tag(text.lower().split())  #nltk.word_tokenize(text)
        lemm_words = [nltk.WordNetLemmatizer().lemmatize(w[0], get_wordnet_pos(w[1])) for w in word_pos]  
        return ' '.join(lemm_words)  
    
    text=stopword(text)
    text=remove(text)
    text=normalize(text)
    return text

data['clean']=data['Description'].apply(processing)




##Save the cleaned version: 
data.to_hdf('text.h5', key='text', mode='w')
datac=pd.read_hdf('text.h5', 'text')
