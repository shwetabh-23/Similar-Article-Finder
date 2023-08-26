from dataloader import Dataloader
from cleaner import Cleaner, remove_stopwords
from utils import topic_modelling, keywords, stemming, lemma, similarity
import pandas as pd
import numpy as np

# Extract the folder and change the below file paths to respective file destinations

src_path = r'Data\train.txt.src'
summary_path = r'Data\train.txt.tgt'

source, summary = Dataloader(src_path, summary_path)
data = pd.DataFrame(data = [source, summary]).T
data.columns = ['article', 'summary']
#Cleaning the data
data['article'] = data['article'].apply(Cleaner)
data['summary'] = data['summary'].apply(Cleaner)

data['article'] = data['article'].apply(remove_stopwords)
data['summary'] = data['summary'].apply(remove_stopwords)

data['topics'] = ((data['article'].apply(topic_modelling) + data['summary'].apply(topic_modelling) + data['article'].apply(keywords) + data['summary'].apply(keywords)))
data['topics'] = data['topics'].apply(lambda x : set(x))
data['topics'] = data['topics'].apply(lambda x : np.array([x for x in x]))
data['topics'] = data['topics'].apply(stemming)
data['topics'] = data['topics'].apply(lemma)

l = 0
data['similar topics'] = 0
all_sim_topics = []
for i in range(len((data['topics']))):
    l1 = data['topics'][i]
    sim_topics = []

    for j in range(len(data['topics'])):
        
        l2 = data['topics'][j]
        if l1 != l2:
            if len(similarity(l1, l2)) > 4:
                sim_topics.append(j)
    all_sim_topics.append(sim_topics)            
data['similar topics'] = (all_sim_topics)           

breakpoint()