import nltk
from nltk import pos_tag , word_tokenize
import re
from nltk.corpus import stopwords

# Run below 3 lines once, then comment it  

# nltk.download("punkt", quiet=True) # tokenizer
# nltk.download("averaged_perceptron_tagger_eng", quiet=True) # for POS
# nltk.download('stopwords')

def clean_stopword_context(phrase, stopwords_set):
    """Removes standard stopwords from an extracted multi-word phrase."""
    words = phrase.split()
    cleaned_words = [word for word in words if word.lower().strip() not in stopwords_set]
    return " ".join(cleaned_words).strip()


def AspectExtraction(text): # Renamed function for clarity
    copulas = {"is", "was", "were", "are", "am", "be", "been", "being"}
    # Global definitions for efficiency and clarity
    NEGATIONS = {"no", "not", "never", "none", "hardly", "barely", "cannot", "can't", "don't", "doesn't", "didn't"}
    CONNECTORS_TO_KEEP = {'and', 'but', 'or', 'nor', 'with', 'for', 'on', 'at', 'to', 'from', 'about', 'by', 'as', 'than'}
    DEFAULT_STOPWORDS = set(stopwords.words("english"))
    STOPWORDS_TO_REMOVE = (DEFAULT_STOPWORDS - NEGATIONS) - CONNECTORS_TO_KEEP

    # --- Preprocessing ---
    text = re.sub(r'([.,])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    aspects = []

    for i, (word, tag) in enumerate(tagged):

        # 1. Multi-Word Aspect Check (NN VB... JJ Pattern)
        if tag.startswith("NN"):
            
            # Build potential multi-word aspect backwards
            aspect_candidate = word
            j = i - 1
            while j >= 0 and (tagged[j][1].startswith("JJ") or tagged[j][1].startswith("NN") or tagged[j][1].startswith("VBG")):
                aspect_candidate = f"{tagged[j][0]} {aspect_candidate}"
                j -= 1
            
            # Check for the NN VB... JJ pattern
            if i + 1 < len(tagged) and tagged[i + 1][0].lower() in copulas:
                
                #  NN VBZ NOT JJ 
                if i + 3 < len(tagged) and tagged[i + 2][0].lower() in NEGATIONS and tagged[i + 3][1].startswith("JJ"):
                    opinion = f"{tagged[i + 2][0]} {tagged[i + 3][0]}"
                    aspects.append((opinion.strip(), aspect_candidate.strip()))
                    continue
                
                #  NN VBZ RB JJ 
                elif i + 3 < len(tagged) and tagged[i + 2][1].startswith("RB") and tagged[i + 3][1].startswith("JJ"):
                    opinion = tagged[i + 3][0] # Only extract the adjective 'good'
                    aspects.append((opinion.strip(), aspect_candidate.strip()))
                    continue
                
                #  NN VBZ JJ (e.g., 'speed is good'). Adjective at i+2.
                elif i + 2 < len(tagged) and tagged[i + 2][1].startswith("JJ"):
                    opinion = tagged[i + 2][0]
                    aspects.append((opinion.strip(), aspect_candidate.strip()))
                    continue
                

            #  Adjective before Noun (JJ NN) 
            if i > 0 and tagged[i - 1][1].startswith("JJ"):
                opinion = tagged[i - 1][0]
                if i - 2 >= 0 and tagged[i - 2][0].lower() in NEGATIONS:
                    opinion = f"{tagged[i - 2][0]} {opinion}"
                aspects.append((opinion.strip(), word.strip()))
            
            #  Noun before Adjective (NN JJ) 
            elif i + 1 < len(tagged) and tagged[i + 1][1].startswith("JJ"):
                opinion = tagged[i + 1][0]
                if i - 1 >= 0 and tagged[i - 1][0].lower() in NEGATIONS:
                    opinion = f"{tagged[i - 1][0]} {opinion}"
                aspects.append((opinion.strip(), word.strip()))

        # VERB/NOUN + ADVERB patterns (VB/NN RB) 
        if tag.startswith(("VB", "NN")):
            if i + 1 < len(tagged) and tagged[i + 1][1].startswith("RB"):
                opinion = tagged[i + 1][0]
                aspect = word
                if i - 1 >= 0 and tagged[i - 1][0].lower() in NEGATIONS:
                    opinion = f"{tagged[i - 1][0]} {opinion}"
                aspects.append((opinion.strip(), aspect.strip()))


    # --- Final Filtering and Cleaning ---
    final_aspects = []
    for op, asp in aspects:
        cleaned_op = clean_stopword_context(op, STOPWORDS_TO_REMOVE)
        cleaned_asp = clean_stopword_context(asp, STOPWORDS_TO_REMOVE)
        
        if cleaned_op and cleaned_asp and cleaned_op.lower() != cleaned_asp.lower():
             final_aspects.append((cleaned_op, cleaned_asp))

    return list(set(final_aspects))
