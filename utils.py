import spacy
import gensim
from gensim import corpora
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from keybert import KeyBERT
import re

def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']): 
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    output = []
    for sent in texts:
        doc = nlp(sent) 
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
    return output

def topic_modelling(text):
  #text_list=df_sampled['Text'].tolist()
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'

    # Split the text into a list of sentences using the regex pattern
    sentences = re.split(sentence_pattern, text)
    tokenized_reviews = lemmatization(sentences)

    dictionary = corpora.Dictionary(tokenized_reviews)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]

    LDA = gensim.models.ldamodel.LdaModel
    topics = []
    # Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=10, iterations = 100)
    words = (lda_model.show_topics(formatted=False, num_words= 5)[0])
    for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
        topics.append(([w[0] for w in topic]))

    return topics[0]

def keywords(text):
    kw_model = KeyBERT()
    return [w[0] for w in (kw_model.extract_keywords(text, top_n = 10))]

def stemming(text):
    y = []
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

def lemma(text):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def similarity(l1, l2):
    return set(l1).intersection(set(l2))