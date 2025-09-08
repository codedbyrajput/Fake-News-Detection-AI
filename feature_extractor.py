"""
Documentation:
In this class I will be working with scikit-learn library. 
After processing the text in textProcessing class, we convert the cleaned text to numbers so that ML model can understand it.
We will find the important words in the text and assign the weightage via TF-IDF (Term Frequency x Inverse Document Frequency)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features = 10000)# save memory and model performance by putting size constraint
        self.is_fit = False

    
    def fit_transform(self, cleaned_documents):
        "Learn vocabulary and return TF-IDF matrix for training set"
        x = self.vectorizer.fit_transform(cleaned_documents)# here .fit filter out the top 10000 unique words and tansform assign them tf idf values using the formula with the help of data collected by .fit
        self.is_fit = True
        return x
    

    "This method is for test data as we dont want to relearn the unique words again. (.fit)"
    def transform(self, cleaned_documents):
        "Transform new cleaned documents into TF-IDF using learned vocab"

        if not self.is_fit:
            raise ValueError("Need to use fit_tranform() method to get a matrix.")
        return self.vectorizer.transform(cleaned_documents)
    
    def get_feature_names(self):
        if not self.is_fit:
            raise ValueError("vectorizer not fitted yet")
        return self.vectorizer.get_feature_names_out()