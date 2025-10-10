import nltk
from nltk import pos_tag , word_tokenize
import re
from nltk.corpus import stopwords

def CleanText(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join([word.lower() for word in text.split() if word.lower() not in stopwords])

def AspectExtraction(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens) # assigning parts of speech to each token in form of tuple (token,POS)
    aspects = []

    for i , (word,tag) in enumerate(tagged):
        if tag.startswith("NN"):
            if i>0 and tagged[i-1][1].startswith("JJ"): # if previous token is adjective
                aspects.append((tagged[i-1][0],word))
            elif i+1 < len(tagged) and tagged[i+1][1].startswith("JJ"):
                aspects.append((word, tagged[i+1][0]))  # (noun, adj)
    return aspects  
    