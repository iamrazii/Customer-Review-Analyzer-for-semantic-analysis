import nltk
from nltk import pos_tag , word_tokenize
import re
from nltk.corpus import stopwords

# Run below 3 lines once, then comment it  
nltk.download("punkt", quiet=True) # tokenizer
nltk.download("averaged_perceptron_tagger_eng", quiet=True) # for POS
nltk.download('stopwords')


def AspectExtraction(text):
    
    NEGATIONS = {"no", "not", "never", "none", "hardly", "barely", "cannot", "can't", "don't", "doesn't", "didn't"}
    CONNECTORS_TO_KEEP = {'and', 'but', 'or', 'nor', 'with', 'for', 'on', 'at', 'to', 'from', 'about', 'by', 'as', 'than'}
    DEFAULT_STOPWORDS = set(stopwords.words("english"))
    STOPWORDS = (DEFAULT_STOPWORDS - NEGATIONS) - CONNECTORS_TO_KEEP 
    # adding spaces between word and adjacent terminal(./,)
    text = re.sub(r'([.,])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip() 
    # Tokenization
    tokens = word_tokenize(text)
    filtered_tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    tagged = pos_tag(filtered_tokens)
    aspects = []

    for i, (word, tag) in enumerate(tagged):
        #  ADJECTIVE + NOUN patterns 
        if tag.startswith("NN"):
            
            # Adjective + Noun (JJ NN)
            if i > 0 and tagged[i - 1][1].startswith("JJ"):
                opinion = tagged[i - 1][0]
                
                # Handle negation 
                if i - 2 >= 0 and tagged[i - 2][0].lower() in NEGATIONS:
                    combined_adj = f"{tagged[i - 2][0]} {opinion}"
                    aspects.append((combined_adj, word))
                else:
                    aspects.append((opinion, word)) 

            # Noun + Adjective (NN JJ)
            elif i + 1 < len(tagged) and tagged[i + 1][1].startswith("JJ"):
                opinion = tagged[i + 1][0]
                aspects.append((opinion, word)) # (Opinion, Aspect)

        # VERB/NOUN + ADVERB pattern (VB/NN RB) ---
        if tag.startswith(("VB", "NN")):
            if i + 1 < len(tagged) and tagged[i + 1][1].startswith("RB"):
                opinion = tagged[i + 1][0]
                aspect = word
                # Handle negation (less common, but robust)
                if i - 1 >= 0 and tagged[i - 1][0].lower() in NEGATIONS:
                    combined_adv = f"{tagged[i - 1][0]} {opinion}"
                    aspects.append((combined_adv, aspect)) 
                else:
                    aspects.append((opinion, aspect))
                
    return aspects