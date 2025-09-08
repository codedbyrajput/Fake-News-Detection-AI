"""
This class will handle the text cleaning and pre processing tasks to transform raw article text into cleaner form

Things to focus:
1. Remove Punctuation, conevrt to lowercase
2. Tokenization
3. Remove Stopwords like and, the, is
4. Lemmatization or Stemming : to reduce work to its base form like balanced to balance (will keep it optional as might add complexity to the project)
5. Extra Whitespace removal after all the work done 
"""
import re
from news_article import NewsArticle
import pandas as pd
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
class TextPreprocessor:

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = PorterStemmer()


    # cleaning the text by removing any unwanted punctuation marks or symbols

    def clean_symbols(self, text):
        text = str(text) # esuring that the text is string 
        modifiedText = re.sub(r'[^a-zA-Z0-9\s]', '' , text)
        return modifiedText

    def to_lower(self, text):
        return text.lower()

    def to_tokens(self, text):
        finalTokens = []
        newText = self.clean_symbols(text)
        textList = newText.split() # converted into a list of string or has been converted to tokens 
        for index in range(len(textList)):
            if(textList[index] not in self.stopwords):
                finalTokens.append(textList[index])
        return finalTokens

    def stemming(self, word):
        return self.stemmer.stem(word)
    
    def clean_text(self, text):
        text = self.to_lower(text)
        tokens = self.to_tokens(text)
        stemmed_tokens = [self.stemming(token) for token in tokens]
        return " ".join(stemmed_tokens).strip()


    



