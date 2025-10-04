import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocessing import ReviewPreprocessor
from genisCore import genisCore
from sentiments import Sentiments

def predict_sentiment(
        sentence: str,
        stars: int,
        model: RandomForestClassifier,
        preprocessor: ReviewPreprocessor):
    """Predict the sentiment score for a given sentence."""

    # Initialize sentiment manager
    sentimentManager = Sentiments()

    pairs: list[tuple[str, str]] = []
    nouns: list[str] = []
    pairs, nouns = genisCore(sentence, preprocessor)

    parsed_scores, _ = sentimentManager.gradeNounSentiment(sentence, nouns)

    # Extract features
    features = [
        stars,
        sum([score for score in parsed_scores.values() if score > 0]),
        sum([score for score in parsed_scores.values() if score < 0]),
        sum([score for score in parsed_scores.values() if score == 0])
    ]

    # Predict sentiment
    prediction = model.predict([features])
    return prediction[0]

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment score for a given sentence.")
    parser.add_argument("sentence", type=str, help="The sentence to analyze.")
    parser.add_argument("stars", type=int, help="The star rating associated with the sentence.")
    parser.add_argument("cache_path", type=str, help="Path to the preprocessing cache directory.")
    args = parser.parse_args()

    # Load the model
    model_path = 'data/random_forest_classifier.pkl'
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        return

    # Initialize the preprocessor
    preprocessor = ReviewPreprocessor(cachePath=args.cache_path)

    # Predict sentiment
    sentiment = predict_sentiment(args.sentence, args.stars, model, preprocessor)
    if sentiment is not None:
        print(f"Predicted sentiment score: {sentiment}")

if __name__ == "__main__":
    main()