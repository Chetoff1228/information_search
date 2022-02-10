import pandas as pd


lyrics = pd.read_csv('lyrics.csv', sep=',', index_col='index')
print(lyrics.head(10))

import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')


with open('popsongs.txt', 'r', encoding='utf-8') as g:
    article_text = g.read()


processed_article = article_text.replace('/n', '')
processed_article = processed_article.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]


from gensim.models import Word2Vec

word2vec = Word2Vec(all_words, min_count=2)


vocabulary = list(word2vec.wv.index_to_key)
print(vocabulary)




