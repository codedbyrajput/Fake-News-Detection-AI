from data_loader import DataLoader
from text_preprocessing import TextPreprocessor
from feature_extractor import FeatureExtractor
from Evaluation import Evaluator
from Interface import PredictionInterface
from fakenews_classifier import FakeNewsClassifier
from sklearn.model_selection import train_test_split
import os

# Change to your project directory
project_path = r"C:\Users\sagar\OneDrive\other stuff\Comp Projects\Fake News AI"
os.chdir(project_path)

print(f"Current working directory: {os.getcwd()}")
print("Files in project directory:")
for file in os.listdir('.'):
    print(f"  {file}")

# 1. Load dataset
articles = DataLoader.load_data()

# Check if we have any articles
if len(articles) == 0:
    print("No articles loaded. Please check your data file.")
    exit()

# 2. Extract text and labels
texts = [article.text for article in articles]
labels = [article.label for article in articles]

# Rest of your code remains the same...
# 3. Preprocess text
preprocessor = TextPreprocessor()
cleaned_texts = [preprocessor.clean_text(text) for text in texts]

# 4. Train/test split
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    cleaned_texts, labels, test_size=0.2, random_state=42
)

# 5. Feature extraction
extractor = FeatureExtractor()
X_train = extractor.fit_transform(X_train_texts)
X_test = extractor.transform(X_test_texts)

# 6. Train model
classifier = FakeNewsClassifier()
classifier.train(X_train, y_train)

# 7. Evaluate
y_pred = classifier.predict(X_test)
metrics = Evaluator.evaluate(y_test, y_pred)

print("Evaluation Results:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# plot confusion matrix
Evaluator.plot_confusion_matrix(y_test, y_pred)

# 8. Save model 
classifier.save_model("fakenews_model.joblib")

# 9. Launch CLI for custom predictions
interface = PredictionInterface(preprocessor, extractor, classifier)
interface.run_cli()