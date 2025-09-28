import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import re
import argparse
import csv
import random
import concurrent.futures

from preprocessing import ReviewPreprocessor
from sentiments import Sentiments
from genisCore import genisCore

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
    for pair in pairs:
        # Each pair is a tuple (adj, noun)
        adj, noun = pair
        # Create a phrase from the adjective and noun
        phrase = f"{adj} {noun}"
        # Get the sentiment scores for the phrase
        score = sid.polarity_scores(phrase)
        # Store the scores in the dictionary
        scores[pair] = score
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

def process_review(
        rawReview: str,
        rawReviewData: dict,
        sentimentsManager: Sentiments,
        preprocessor: ReviewPreprocessor) -> tuple[str, dict, str]:
    """
    Process a review to extract its sentiment scores using a LLM.
    Args:
        rawReview (str): The original review text.
        rawReviewData (dict): The preprocessed review data.
        sentimentsManager (Sentiments): An instance of the Sentiments class.
        preprocessor (ReviewPreprocessor): An instance of the ReviewPreprocessor class.
    Returns:
        tuple: A tuple containing the original review, parsed scores, and state.
        State can be one of the following:
            - "C": Cached data used
            - "E": Error occurred during parsing
            - "J": JSON error occurred
            - "_": Successfully parsed
    """
    state: str = ""
    parsed_scores: dict = {}

    cachedReview = preprocessor.GetReviewFromCache(rawReview)
    if cachedReview is not None and "parsed_scores" in cachedReview:
        # Use the cached data
        state = "C"
        parsed_scores = cachedReview["parsed_scores"]
    else:
        readableReview: str = rawReviewData["readable"]
        pairs: list[tuple[str, str, dict]] = rawReviewData["pairs"]
        nouns: list[str] = pairs and [noun for _, noun, _ in pairs]
        parsed_scores, state = sentimentsManager.gradeNounSentiment(
            readableReview,
            nouns,
        )
        if state in ("E", "J"):
            cachedReview = None
        else:
            state = "_"
            preprocessor.AddSubitemsToReviewCache(rawReview, {"parsed_scores": parsed_scores})
    return rawReview, parsed_scores, state

def process_grade(
        rawReview: str,
        reviewData: dict,
        sentimentsManager: Sentiments,
        preprocessor: ReviewPreprocessor
    ) -> tuple[str, float, str]:
    """
    Process a review to extract its sentiment score using a LLM.
    Args:
        rawReview (str): The original review text.
        reviewData (dict): The preprocessed review data.
        sentimentsManager (Sentiments): An instance of the Sentiments class.
        preprocessor (ReviewPreprocessor): An instance of the ReviewPreprocessor class.
    Returns:
        tuple: A tuple containing the original review, grade, and state.
        State can be one of the following:
            - "C": Cached data used
            - "E": Error occurred during parsing
            - "_": Successfully parsed
    """
    cachedReview = preprocessor.GetReviewFromCache(rawReview)
    if cachedReview is not None and "LLM-score" in cachedReview and cachedReview["LLM-score"] != 0:
        # Use the cached data if it exists and is not zero.
        # A value of zero means that the review was not processed, in most cases
        # because the LLM returned an error.
        return rawReview, cachedReview["LLM-score"], "C"
    else:
        grade, state = sentimentsManager.gradeReview(reviewData["readable"])
        preprocessor.AddSubitemsToReviewCache(rawReview, {"LLM-score": grade})
        return rawReview, grade, state

def main():
    ver: str = "0.14.1"
    # Labels for the text and rating in the jsonl file
    # The default values are the ones used in the Amazon reviews dataset
    label_text: str = "text"
    label_rating: str = "rating"

    # Create an instance of classes used in the script.
    # Initialization postponed as it requires the cache path, calculated later.
    sentimentsManager: Sentiments
    preprocessor: ReviewPreprocessor

    print(f"GENIS calc v{ver}")
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Jsonl file to process including extension")
    parser.add_argument("-s", "--seed", type=int, help="Random seed (default 1967)", default=1967)
    parser.add_argument("-m", "--max-reviews", type=int, help="Reviews to process (default 1000)", default=1000)
    parser.add_argument("-v", "--version", action="version", version=f"{ver}")
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

    sentimentsManager = Sentiments()
    preprocessor = ReviewPreprocessor(cachePath = topicPath)

    print(f"""This run uses random seed {seed}
Files are stored in the following structure (see README.md for details):

data
|-- {topicGeneral}
    |-- correction_cache.json       (spelling correction cache)
    |-- preprocessing_cache.json    (intermediate results)
    |-- {seed}
        |-- selected_reviews.csv    (for human grading)
        
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

    print(f"Processing...")
    for index, (rawReview, rawReviewData) in enumerate(preprocessed_reviews.items()):

        if (index and (index % 100 == 0)):
            print(f"{index} of {len(preprocessed_reviews)}", flush=True)

        # Check if the data we are about to calculate are already in the cache.
        # If so, skip the calculation.
        cachedReview = preprocessor.GetReviewFromCache(rawReview)
#        if cachedReview is not None and ("pairs" in cachedReview) and ("nouns" in cachedReview):
        if cachedReview is not None and ("pairs" in cachedReview):
            pass
        else:        
            # If not, process the review to extract adjective-noun pairs.
            # split sentences on hard punctuation (periods, exclamation marks, question marks)
            pairs: list[tuple[str, str]] = []
            nouns: list[str] = []
            pairs, nouns = genisCore(rawReviewData["corrected"], preprocessor, nlp)

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
        if (index and index % 100) == 0:
            print(f"{index} of {len(reviews_dict)}")
        index += 1

        cachedReview = preprocessor.GetReviewFromCache(rawReview)
        if cachedReview is not None:
            if "pairs" in cachedReview:
                if cachedReview["pairs"] != []:
                    # Use the cached data but only if it is not empty.
                    # If "pairs" is defined, "nouns" is necessarily defined too,
                    # so this is the only check we need to do.
                    filtered_pairs = cachedReview["pairs"]
                else:
                    continue
            else:
                filtered_pairs: list[tuple[str, str, dict]] = []
                pairs = rawReviewData["pairs"]
                # Calculate the sentiment scores for the pairs, then filter out
                # those with a compound score below 0.05
                pairsAndScores = analyze_sentiment(pairs = pairs, sid = sid)
                for pair in pairsAndScores:
                    scores = pairsAndScores[pair]
                    if abs(scores['compound']) >= 0.05:
                        filtered_pairs.append((pair[1], pair[0], scores))
                preprocessor.AddSubitemsToReviewCache(rawReview, {"pairs": filtered_pairs})
            """
            if "nouns" in cachedReview:
                # Use the cached nouns
                filtered_nouns = cachedReview["nouns"]
                # Also try to use cached VADER scores for nouns if available
                if "nouns_vader_scores" in cachedReview:
                    filtered_nouns_vader_scores = cachedReview["nouns_vader_scores"]
                else:
                    # Calculate VADER scores for each noun
                    filtered_nouns_vader_scores = {noun: sid.polarity_scores(noun) for noun in filtered_nouns}
                    preprocessor.AddSubitemsToReviewCache(rawReview, {"nouns_vader_scores": filtered_nouns_vader_scores})
            else:
                filtered_nouns = sorted(list(set([noun for noun, _ in filtered_pairs])))
                preprocessor.AddSubitemsToReviewCache(rawReview, {"nouns": filtered_nouns})
                # Calculate VADER scores for each noun
                filtered_nouns_vader_scores = {noun: sid.polarity_scores(noun) for noun in filtered_nouns}
                preprocessor.AddSubitemsToReviewCache(rawReview, {"nouns_vader_scores": filtered_nouns_vader_scores})
            """
            if "V-Whole" in cachedReview:
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
#            "nouns": filtered_nouns,
            "O-Score": rawReviewData["O-Score"],
            "V-Whole": V_Whole,
        }

    #
    # In this last step, we invoke a LLM to parse the sentiment score.
    # In fact, we ask to tasks to the LLM:
    # 1. Parse the review text and the noun list, and return a score.
    # 2. Assign directly a score to the review text using a zero-shot approach.
    #
    print("\nCalculating LLM scores for the reviews.")
    print("Approach 1: LLM scores for each relevant noun.")
    print("_ = calculated, C = cached, E = error, J = json error")
    # Use a ThreadPoolExecutor to call parseScore in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_review, rawReview, rawReviewData, sentimentsManager, preprocessor): rawReview 
            for rawReview, rawReviewData in filtered_reviews_dict.items()
        }
        index = 0
        for future in concurrent.futures.as_completed(futures):
            rawReview, parsed_scores, state = future.result()
            print(f"{state}", end="", flush=True)
            index += 1
            if (index % 100) == 0:
                print(f" {index} of {len(filtered_reviews_dict)}", flush=True)
            if state == "E" or state == "J":
                # If there was an error, there is nothing to do.
                continue
            # Calculate scores using parsed_scores
            plusValues = sum([score for score in parsed_scores.values() if score > 0])
            minusValues = sum([score for score in parsed_scores.values() if score < 0])
            neutralValues = sum([score for score in parsed_scores.values() if score == 0])
            methodScore = sum(parsed_scores.values())
            # Update the review data with the LLM score
            filtered_reviews_dict[rawReview]["G-score"] = methodScore
            filtered_reviews_dict[rawReview]["G-scoreP"] = plusValues
            filtered_reviews_dict[rawReview]["G-scoreM"] = minusValues
            filtered_reviews_dict[rawReview]["G-scoreN"] = neutralValues
    print("")

    print("\nApproach 2: LLM zero-shot score for the review.")
    print("_ = calculated, C = cached, E = error")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures_grade = {
            executor.submit(process_grade, rawReview, rawReviewData, sentimentsManager, preprocessor): rawReview 
            for rawReview, rawReviewData in filtered_reviews_dict.items()
        }
        index = 0
        for future in concurrent.futures.as_completed(futures_grade):
            rawReview, grade, state = future.result()
            index += 1
            print(f"{state}", end="", flush=True)
            if (index % 100) == 0:
                print(f" {index} of {len(filtered_reviews_dict)}", flush=True)
            if state == "E":
                grade = 0
            filtered_reviews_dict[rawReview]["LLM-score"] = grade
    print("")

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

    print("\nDone.")

if __name__ == "__main__":
    main()
