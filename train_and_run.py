import os
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from text_preprocessing import TextPreprocessor
from feature_extractor import FeatureExtractor
from Evaluation import Evaluator
from Interface import PredictionInterface
from fakenews_classifier import FakeNewsClassifier
import random


def main() -> None:
    project_path = r"C:\Users\sagar\OneDrive\other stuff\Comp Projects\Fake News AI"
    os.chdir(project_path)

    print(f"Current working directory: {os.getcwd()}")

    # 1) Load dataset
    articles = DataLoader.load_data()
    
    sample = random.sample(articles, 5)
    for a in sample:
        print("\n--- SAMPLE ---")
        print("TRUE:", "FAKE" if a.label == 0 else "REAL")
        print("TITLE:", a.title)
    if not articles:
        print("ERROR: No articles loaded. Check your dataset file.")
        return

    # 2) Extract title + text + labels
    texts = [f"{a.title} {a.text}" for a in articles]
    labels = [a.label for a in articles]   # should be 0/1 from DataLoader


    # 3) Preprocess
    preprocessor = TextPreprocessor()
    cleaned_texts = [preprocessor.clean_text(t) for t in texts]

    # 4) Split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42
    )

    # 5) Features
    extractor = FeatureExtractor()
    X_train = extractor.fit_transform(X_train_texts)
    X_test = extractor.transform(X_test_texts)

    # 6) Train
    classifier = FakeNewsClassifier()
    classifier.train(X_train, y_train)

    # 7) Evaluate
    y_pred = classifier.predict(X_test)
    metrics = Evaluator.evaluate(y_test, y_pred)

    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    Evaluator.plot_confusion_matrix(y_test, y_pred)

    # 8) Save model
    classifier.save_model("fakenews_model.joblib")

    # 9) CLI
    PredictionInterface(preprocessor, extractor, classifier).run_cli()


if __name__ == "__main__":
    main()
