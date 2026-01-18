# Fake-News-Detection-AI Model
The Fake News Detection AI Model is a machine learning project built to classify news articles as real or fake using Natural Language Processing (NLP).

This project highlights the use of text preprocessing, feature extraction, and supervised learning to address misinformation and fake news detection.

# Features
Preprocessing pipeline: text cleaning, stopword removal, lemmatization.  
Feature extraction: Bag-of-Words (CountVectorizer) & TF-IDF representation.  
ML classifier: Logistic Regression for binary classification.  
High accuracy on labeled dataset of real & fake news.  
Console-based prediction: Input a headline or article, get instant results.  

# Tech Stack
Python 3  
scikit-learn – model training and evaluation
pandas, numpy – dataset handling  
nltk / re – preprocessing text  
joblib – saving/loading trained model  

# Model Workflow
Load dataset of real vs fake news.  
Preprocess text (cleaning, tokenization, stopwords removal).  
Convert text into numerical features (TF-IDF).  
Train Logistic Regression classifier.  
Evaluate accuracy, precision, recall, and F1-score.  
Save trained model with joblib for reuse.

# Future Improvements
Add more classifiers (Naïve Bayes, Random Forest, Deep Learning).  
Build a web app with Flask/Django for UI-based prediction.  
Expand dataset for multilingual news detection.  
Deploy the model on cloud for public use.

# Output Screenshots

# Licence
This project is licensed under the MIT License.

