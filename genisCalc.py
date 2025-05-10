import os
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import re
import argparse
import csv
import random

from preprocessing import ReviewPreprocessor
from embeddings import EmbeddingsManager
from sentiments import Sentiments

# Define the analyze_sentiment function
def analyze_sentiment(pairs: tuple[str, str],sid = None) -> dict[str, dict]:
    """
    Analyze sentiment for a list of noun/adjective pairs using VADER.
    Args:
        pairs (list of tuples): List of tuples containing noun/adjective pairs.
        sid (SentimentIntensityAnalyzer): Optional VADER sentiment analyzer instance.
        If not provided, a new instance will be created.
    Returns:
        dict: A dictionary with sentiment scores for each pair, using VADER
        format (compound, pos, neg, neu).
    """
    if sid is None:
        sid = SentimentIntensityAnalyzer()
    scores: dict[str, dict] = {}

    for noun, adj in pairs:
        phrase = f"{adj} {noun}"
        score = sid.polarity_scores(phrase)
        scores[phrase] = score
    return scores

def cbDotPrint(index: int, total: int) -> None:
    """
    Callback function to print progress.
    Args:
        index (int): Current index.
        total (int): Total number of items.
    """
    if (index and (index % 100)) == 0:
        print(f"{index} of {total}",flush=True)
#        print(".",end="",flush=True)
    if index == total:
        print("Done.",flush=True)

def main():
    ver: str = "0.9.0"
    # Labels for the text and rating in the jsonl file
    # The default values are the ones used in the Amazon reviews dataset
    label_text: str = "text"
    label_rating: str = "rating"

    # Create an instance of classes used in the script.
    # Initialization postponed as it requires the cache path, calculated later.
    sentimentsManager: Sentiments = None
    embeddingManager: EmbeddingsManager = None
    preprocessor: ReviewPreprocessor = None

    print(f"Amazon Cluster Analysis v{ver}")
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Jsonl file to process including extension")
    parser.add_argument("-s", "--seed", type=int, help="Random seed (default 1967)", default=1967)
    parser.add_argument("-m", "--max-reviews", type=int, help="Reviews to process (default 1000)", default=1000)
    parser.add_argument("-v", "--version", action="version", version=f"{ver}")
    parser.add_argument("-r", "--runs", type=int, help="Number of runs (default 1)", default=1)
    parser.add_argument("-n", "--noimages", action="store_true", help="Do not show images, useful for batch processing")
    parser.add_argument("-fr", "--forcerandom", action="store_true", help="Use random scores instead of sentiment analysis with LLM")
    args = parser.parse_args()

    # Process command line arguments
    # Add the .jsonl extension if not present
    if not args.filename.endswith(".jsonl"):
        args.filename = os.path.splitext(args.filename)[0] + ".jsonl"
    if not os.path.exists(args.filename):
        print(f"File {args.filename} not found.")
        return
    seed = args.seed
    reviewsFileName = args.filename

    # Calculate paths
    topicGeneral = os.path.splitext(os.path.basename(reviewsFileName))[0]
    topicPath = os.path.join("data", topicGeneral)
    if not os.path.exists(topicPath):
        os.makedirs(topicPath)
    # Create a directory for the topic and seed
    topicSeedPath = os.path.join(topicPath, str(seed))
    if not os.path.exists(topicSeedPath):
        os.makedirs(topicSeedPath)

    # Instantiate the embeddings manager
    #embeddingManager = EmbeddingsManager(cachePath = topicPath)
    # Initialize sentiment cache using the instance method
    sentimentsManager = Sentiments(cachePath=topicPath)
    # Instantiate the ReviewPreprocessor class (which now handles cache initialization)
    preprocessor = ReviewPreprocessor(cachePath = topicPath)    # Set result file (csv format, containing original and adjusted ratings)

    result_file = os.path.join(topicGeneral, f"{topicGeneral}_results.csv")

    print(f"""This run uses random seed {seed}
Files are stored in the following structure (see README.md for details):

data
|-- {topicGeneral}
    |-- correction_cache.json
    |-- embeddings_cache.pkl
    |-- preprocessing_cache.json
    |-- {seed}
        |-- sentiments_cache.json
        |-- results.csv             (training and prediction results, available only after ML training) 
        |-- selected_reviews.csv    (for human grading)
        |-- ML_model.pkl            (trained ML model)
        
"""
    )

    # Load the SpaCy model
    print("Loading SpaCy model...")
    nlp = spacy.load("en_core_web_sm")
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Load a random sample of reviews from the file.
    print(f"Loading a subset of {args.max_reviews} reviews from {reviewsFileName}...")
    original_reviews, original_indices = preprocessor.LoadReviews(
        file_path=reviewsFileName,
        max_reviews=args.max_reviews,
        label_text=label_text, label_rating=label_rating,
        seed=seed, callback=cbDotPrint
    )
    print(f"\nLoaded {len(original_reviews)} reviews.")

    print("Preprocessing reviews...")
    # Build a dict mapping review text to its rating (this is useful for caching).
    reviews_dict: dict[str, float] = {
        str(review[label_text]): review[label_rating] for review in original_reviews
    }
    # Call the class method on the preprocessor instance.
    preprocessed_reviews = preprocessor.PreprocessReviews(reviews_dict,callback=cbDotPrint)
    del original_reviews,original_indices

    # Starting from this point, the code is repeated for each run
    for run in range(args.runs):
        #
        # First phase: extract adjective-noun pairs from the reviews.
        #
        pairs: list[tuple[str, str]] = []
        nouns: list[str] = []
        # The dictionary associates the original review with its corrected form and
        # the adjective-noun pairs.
        reviews_dict: dict[tuple[str,str, str]] = {}
        # The dictionary associates the original review with its corrected form and
        # the adjective-noun pairs which VADER considers relevant (abs(compound) >= 0.05).
        filtered_reviews_dict: dict[str, list[dict]] = {}
        # The dictionary associates a review with the sentiment scores that the LLM
        # calculated for each noun in filtered_reviews_dict(review).
        parsed_scores: dict[str, dict] = {}

        for rawReview, rawReviewData in preprocessed_reviews.items():

            # Check if the data we are about to calculate are already in the cache.
            # If so, skip the calculation.
            cachedReview = preprocessor.GetReviewFromCache(rawReview)
            if cachedReview is not None and ("pairs" in cachedReview) and ("nouns" in cachedReview):
                pass
            else:
                # If not, process the review to extract adjective-noun pairs.
                # split sentences on hard punctuation (periods, exclamation marks, question marks)
                pairs = []
                nouns = []
                # This regex splits on periods, exclamation marks, and question marks,
                sentences = re.split(r'(?<=[.!?]) +', rawReviewData["corrected"])
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 4:
                        continue
                    # Process the sentence with SpaCy.
                    # This is the core idea of the method: we assume that the sentiment in a review
                    # is mainly expressed by nouns combined with adjectives, like in "good music"
                    # or "awful service"
                    # The extraction uses Spacy.
                    # - "amod" means adjectival modifier (e.g., "good" in "good music")
                    # - "acomp" means adjectival complement (e.g., "good" in "the product is good")
                    # - "nsubj" means nominal subject (e.g., "product" in "the product is good") 
                    doc = nlp(sentence)
                    for token in doc:
                        if token.pos_ == "NOUN":
                            # Token "children" are the words that depend on it.
                            for child in token.children:
                                if child.dep_ == "amod":
                                    # adjective modifier (e.g., "good" in "good music")
                                    pairs.append((token.text, child.text))
                                    nouns.append(token.text)
                        elif token.dep_ == "acomp":
                            # adjectival complement (e.g., "good" in "the product is good").
                            # Now search its subject (the noun).
                            subjects = [child for child in token.head.children if child.dep_ == "nsubj"]
                            if subjects:
                                # Found, we can add the pair
                                pairs.append((subjects[0].text, token.text))
                                nouns.append(subjects[0].text)

                # Lemmatization is useful for cases where singual and plural forms are used
                # interchangeably, like "good music" and "good musics".
                pairs = [(preprocessor.LemmatizeText(noun), adj) for noun, adj in pairs]
                # Remove duplicates from pairs
                pairs = sorted(list(set(pairs)))
                # Recalculate the nouns based on the pairs
                nouns = sorted(list(set([noun for noun, _ in pairs])))

            # Add the pairs to the review_dict for later sentiment analysis.
            # Differently, the review_dict uses the corrected review text as the key.
            reviews_dict[rawReview] = {
                "O-Score": rawReviewData["score"],
                "readable": rawReviewData["readable"],
                "corrected": rawReviewData["corrected"],
                "nouns": nouns,
                "pairs": pairs
            }

        #
        # Second phase: calculate the sentiment scores for the reviews.
        #
        index: int = 0
        print("\nCalculating sentiment scores for the reviews...")
        for rawReview, rawReviewData in reviews_dict.items():
            index += 1
            if (index % 100) == 0:
                print(f"{index} of {len(reviews_dict)}")

            cachedReview = preprocessor.GetReviewFromCache(rawReview)
            if cachedReview is not None:
                if "pairs" in cachedReview:
                    # Use the cached data
                    filtered_pairs = cachedReview["pairs"]
                else:
                    pairs = rawReviewData["pairs"]
                    # Calculate the sentiment scores for the pairs, then filter out
                    # those with a compound score below 0.05
                    scores = analyze_sentiment(pairs = pairs, sid = sid)
                    filtered_pairs = [
                        (pair.split()[1], pair.split()[0]) 
                        for pair, score in scores.items()
                        if abs(score['compound']) >= 0.05
                    ]
                    # Skip if no pair meets the criteria
                    if not filtered_pairs:
                        continue
                if "nouns" in cachedReview:
                    # Use the cached nouns
                    filtered_nouns = cachedReview["nouns"]
                else:
                    filtered_nouns = sorted(list(set([noun for noun, _ in filtered_pairs])))

                # Also update the cache, as the pairs and nouns may have changed.
                # We are not interested in storing the scores.
                preprocessor.AddSubitemsToReviewCache(rawReview, {"pairs": filtered_pairs})
                preprocessor.AddSubitemsToReviewCache(rawReview, {"nouns": filtered_nouns})

            if cachedReview is not None and "V-Whole" in cachedReview:
                V_Whole = cachedReview["V-Whole"]
            else:
                V_Whole = sid.polarity_scores(rawReview)["compound"]
                preprocessor.AddSubitemsToReviewCache(rawReview, {"V-Whole": V_Whole}) 

            # Add a new key to the filtered_reviews_dict dictionary. We also store
            # compound, it will be used later.
            filtered_reviews_dict[rawReview] = {
                "readable": rawReviewData["readable"],
                "corrected": rawReviewData["corrected"],
                "pairs": filtered_pairs,
                "nouns": filtered_nouns,
                "O-Score": rawReviewData["O-Score"],
                "V-Whole": V_Whole,
            }

        print(f"{len(filtered_reviews_dict)} of {len(reviews_dict)} have relevant sentiments.")

        #
        # In this last step, we invoke a LLM to parse the sentiment score.
        # In fact, we ask to tasks to the LLM:
        # 1. Parse the review text and the noun list, and return a score.
        # 2. Assign directly a score to the review text using a zero-shot approach.
        #
        index = 0
        print("\nCalculating LLM scores for the reviews.")
        # Iterate through each review in the filtered_reviews_dict
        print("Approach 1: LLM parsing of score for each relevant noun.")
        print("_ = calculated, C = cached, E = error, J = json error")
        for rawReview, rawReviewData in filtered_reviews_dict.items():
            # Invoke parseScore() and store the result in the dictionary
            # If the data is already in the cache, use it.
            cachedReview = preprocessor.GetReviewFromCache(rawReview)
            if cachedReview is not None and "parsed_scores" in cachedReview:
                # Use the cached data
                parsed_scores = cachedReview["parsed_scores"]
                state = "C"
            else:
                parsed_scores[rawReview], state = sentimentsManager.parseScore(rawReviewData["readable"], rawReviewData["nouns"])
                # If the state is "E" or "J", skip the review and continue to the next one.
                if state == "E" or state == "J":
                    # Add the review to the filtered_reviews_dict with a score of 0.
                    filtered_reviews_dict[rawReview]["L-score"] = 0
                    filtered_reviews_dict[rawReview]["L-scoreP"] = 0
                    filtered_reviews_dict[rawReview]["L-scoreM"] = 0
                    filtered_reviews_dict[rawReview]["L-scoreN"] = 0
                    # Add the parsed scores to the cache
                    preprocessor.AddSubitemsToReviewCache(rawReview, {"parsed_scores": {}})
                    continue
                else:
                    # Add the parsed scores to the cache
                    preprocessor.AddSubitemsToReviewCache(rawReview, {"parsed_scores": parsed_scores[rawReview]})
            print(f"{state}", end="")
            index += 1
            if (index and (index % 100) == 0):
                print(f" {index} of {len(filtered_reviews_dict)}")
            # Calculate the LLM score as the sum of the parsed scores, then
            # add it to the review dictionary.
            plusValues = sum([score for score in parsed_scores[rawReview].values() if score > 0])
            minusValues = sum([score for score in parsed_scores[rawReview].values() if score < 0])
            neutralValues = sum([score for score in parsed_scores[rawReview].values() if score == 0])
            methodScore = sum(parsed_scores[rawReview].values())
            # Update the review dictionary with the LLM score
            rawReviewData["L-score"] = methodScore
            rawReviewData["L-scoreP"] = plusValues
            rawReviewData["L-scoreM"] = minusValues
            rawReviewData["L-scoreN"] = neutralValues

        print("\nApproach 2: LLM zero-shot score for the review.")
        print("_ = calculated, C = cached, E = error")
        index = 0
        for rawReview, rawReviewData in filtered_reviews_dict.items():
            if preprocessor.GetReviewFromCache(rawReview) is not None and "LLM-score" in preprocessor.GetReviewFromCache(rawReview):
                # Use the cached data
                grade = preprocessor.GetReviewFromCache(rawReview)["LLM-score"] 
                state = "C"
            else:
                grade, state = sentimentsManager.assignGradeToReview(rawReviewData["readable"])
                preprocessor.AddSubitemsToReviewCache(rawReview, {"LLM-score": grade})
            print(f"{state}", end="")
            index += 1
            if (index and (index % 100) == 0):
                print(f" {index} of {len(filtered_reviews_dict)}")
            # If the state is "E", skip the review and continue to the next one.
            if state == "E":
                grade = 0
            rawReviewData["LLM-score"] = grade
        print("\nDone.")

        # Save a subset of 100 reviews to a CSV file for human grading and further processing.
        # Human scores must be added manually, so there is the risk of overwriting the file
        # if it already exists. The user is prompted to overwrite the file or not.

        # Select up to 100 reviews
        num_reviews_to_select: int = min(100, len(filtered_reviews_dict))
        writer: None
        fieldnames: list[str] = ['readable','hscore','review']

        random.seed(seed)
        selected_reviews = random.sample(list(filtered_reviews_dict.items()), num_reviews_to_select)
        # Save the selected reviews to a csv file
        selected_reviews_file = os.path.join(topicSeedPath, f"selected_reviews.csv")
        # Check if the file already exists and ask the user if they want to overwrite it
        canWrite: bool = True
        if os.path.exists(selected_reviews_file):
            canWrite = input(f"{selected_reviews_file} already exists. Overwrite it? (y/n): ").lower() == 'y'
        if canWrite:
            with open(selected_reviews_file, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for rawReview, rawReviewData in selected_reviews:
                    # the sole purpose of saving 'review' is that it is used
                    # as dictionary key in the next step.
                    writer.writerow({
                        'readable': f"{rawReviewData['readable']}",
                        'hscore': 0,
                        'review': rawReview
                    })

    print(f"\nSelected reviews saved in {selected_reviews_file}.")
    print(f"Please add the human scores to {selected_reviews_file}, then run script genisTrain.py.")

if __name__ == "__main__":
    main()
