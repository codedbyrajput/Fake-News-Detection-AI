"""
In this class I will be using logistic regression for classification of fake and real news using the scikit-learn. If something goes wrong we can use custom ModelTrainingException. 
"""

from sklearn.linear_model import LogisticRegression
from myExceptions import ModelTrainingException

class FakeNewsClassifier:

    #constructor
    def __init__(self):
        self.model = LogisticRegression(max_iter = 1000, random_state = 10)#random_state acts like a seed
        self.isTrained = False

    # The method parameters: X - TF-IDF matrix that we generated in feature extractor class and y is the label of true or false for the news data
    def train(self, X, y):
        if X is None or y is None or len(y) == 0:
            raise ModelTrainingException("Training data incomplete")
        
        try:
            self.model.fit(X , y)
        except Exception as e:
            raise ModelTrainingException(f"Training failed: {e}")
        self.isTrained = True

# X is the TF-IDF matrix of new articles. This method is used to prodict labels as fake or real by using the trained model. This will return an array of 0 and 1.
    def predict(self, X):
        if not self.isTrained:
            raise ModelTrainingException()
        return self.model.predict(X)
    
    # This works same as predict but returns the probabilities instead of labels. This will return probabilities of the fake news 
    def predict_proba(self, X):
        if not self.isTrained:
            raise ModelTrainingException("Model not trained yet.")
        proba = self.model.predict_proba(X)
        # Make sure we return P(FAKE). FAKE is label 0.
        idx_fake = list(self.model.classes_).index(0)
        return proba[:, idx_fake]



    # I am writing this method to save my trained model otherwise i need to train it everytime i run the code

    def save_model(self, path):
        from joblib import dump
        dump(self.model, path)

    # This saves the whole class as a final trained model 
    @classmethod
    def load_model(cls, path):
        from joblib import load
        clf = cls()
        clf.model = load(path)
        clf.isTrained = True
        return clf