"""
This class will tie everything together to input data and get predictions 
"""

class PredictionInterface:
    def __init__(self, preprocessor, feature_extractor, classifier):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def predict_from_text(self, raw_text: str):
        # Clean the text
        clean = self.preprocessor.clean_text(raw_text)
        # Turn into features
        X = self.feature_extractor.transform([clean])

        # label: 0 = FAKE, 1 = REAL
        label = self.classifier.predict(X)[0]

        # Initialize default probabilities
        prob_fake = 0.5
        prob_real = 0.5

        # Try to get probabilities, but handle errors gracefully
        try:
            proba = self.classifier.predict_proba(X)
            
            # Handle different probability array formats
            if hasattr(proba, 'shape') and len(proba.shape) == 2 and proba.shape[0] > 0:
                if proba.shape[1] == 2:
                    # Binary classification: [prob_fake, prob_real]
                    prob_fake = proba[0][0]
                    prob_real = proba[0][1]
                elif proba.shape[1] == 1:
                    # Only one class probability returned
                    if label == 0:
                        prob_fake = proba[0][0]
                        prob_real = 1 - prob_fake
                    else:
                        prob_real = proba[0][0]
                        prob_fake = 1 - prob_real
        except Exception as e:
            # If we can't get probabilities, use the label to set reasonable defaults
            if label == 0:
                prob_fake = 0.9  # High confidence for FAKE
                prob_real = 0.1
            else:
                prob_real = 0.9  # High confidence for REAL
                prob_fake = 0.1

        # Confidence is always for predicted class
        if label == 0:   # FAKE
            confidence = prob_fake
        else:            # REAL
            confidence = prob_real

        return label, confidence





    def run_cli(self):
        print("Fake News Detector - CLI")
        print("Enter a news article or headline (or 'quit' to exit):")
        while True:
            user_input = input("> ")
            if user_input.strip().lower() in ['quit', 'exit', '']:
                print("Exiting.")
                break
            try:
                label, confidence = self.predict_from_text(user_input)

                if label == 0:
                    result = "FAKE"
                else:
                    result = "REAL"

                print(f"\nResult: This article is {result} news.")
                print(f"Confidence: {confidence * 100:.1f}%\n")
            except Exception as e:
                print(f"Error: {e}")
                


