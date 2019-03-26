# (2-1) LDA on others; 

#############
sample=others 
# model prep;
texts_clean=[i.split()  for i in sample['lossdespt_c2']]
id2word = corpora.Dictionary(texts_clean)

#wrd-freq map;
corpus = [id2word.doc2bow(text) for text in texts_clean]

#id2wrod; 
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]) 

%time
lda = gensim.models.ldamodel.LdaModel(corpus=corpus,  id2word=id2word,  num_topics=11,
                                      random_state=100, update_every=10, chunksize=1000,
                                      passes=100, alpha='auto', per_word_topics=True) 
topics_matrix = lda.show_topics(formatted=False, num_words=10) #topic v.s. top 10 words (can tweak around 10);

#print_topics; 
lda.print_topics(15) 
all_topics = lda.get_document_topics(corpus[:], per_word_topics=True)

doc_topic = [] 
for topic in all_topics:
    doc_topic.append(topic[0])
sample['doc_topics'] = doc_topic 
sample['topic_num'] = [max(doc_top, key=lambda item: item[1])[0] for doc_top in doc_topic]  
def cate(num):
    if num==2:
        return 'water'
    elif num==3:
        return 'theft'
    elif num==6:
        return 'vehicle_damage'
    elif num==8:
        return 'wind' 
    elif num==10:
        return 'lightning_'
    else:
        return 'others'
sample['topic']=sample['topic_num'].apply(cate)
