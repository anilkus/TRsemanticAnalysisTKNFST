import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import fitz
import bs4 as bs
import urllib.request
import re
import nltk


with open("original_abstract.txt", 'r', encoding="utf-8") as text2:
    abstract = text2.read()
with open("text.txt", 'r', encoding="utf-8") as text:
    text = text.read()
      
 #Preprocessing   
abstract=abstract.lower()
abstract = re.sub(r'\([^()]*\d+[^()]*\)', '', abstract)
abstract = re.sub(r'\[[^\[\]]*\d+[^\[\]]*\]', '', abstract)
#text = re.sub(r'\[[0-9]*\]', ' ', text)
#text = re.sub(r'\s+', ' ', text)  
text = text.lower()
#text = re.sub(r'\s+', ' ', text)
#text = re.sub(r'\([^)]*\)', '', text)
#text = re.sub(r'/',' ',text)
#text = re.sub("(\d+)","",text) 
text = re.sub(r'\([^()]*\d+[^()]*\)', '', text)
text = re.sub(r'\[[^\[\]]*\d+[^\[\]]*\]', '', text)
text = text.replace("ark.", '')

stopwords=["Tablo","tablo","ark.","ark","ya","/","(1)"]
print(stopwords)

text = re.sub(r'\b(' + '|'.join(stopwords) + r')\b', '', text)

    
def text_summarizer(text, ratio=0.4):

    sent_list = sent_tokenize(text)
    word_list = " ".join(sent_list).split()
    word_freq = dict(nltk.FreqDist(word_list))
    G = nx.Graph()
    for word in word_freq:
        G.add_node(word, weight=word_freq[word])
    rank = nx.pagerank(G, alpha=0.85)    
    summarize_text = []
    for i in sorted(rank, key=rank.get, reverse=True):
        summarize_text.append(i)
    return " ".join(summarize_text[:int(len(summarize_text) * ratio)])

#TextRank
print(text_summarizer(text))
summary=text_summarizer(text)
from rouge import Rouge
ROUGE = Rouge()

print(ROUGE.get_scores(summary, abstract))

print("orjinal özet: ", len(abstract))
print("orjinal metin: ", len(text))
print("olusturulan özet: ", len(summary))

gercekmetinuzunlugu=len(text)
özetuzunlugu=len(summary)
orjinalozet=len(abstract)


#save your summary
ozet = summary
with open("target_path.txt", "w",encoding="utf-8") as file:
    file.write(ozet)
