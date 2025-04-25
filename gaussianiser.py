import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

from preprocessing import ReviewPreprocessor
from embeddings import EmbeddingsManager

import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sentiments import (
    sentimentCache_init,
    sentimentCache_getSentimentAndAdjustedRating,
    sentiment_aggregateSimilarTopics,
    sentiment_returnMostRelevantTopic,
    sentiment_getTypicalTopics,
    sentiment_topicsFromSentence,
)

embeddingManager: EmbeddingsManager = None
preprocessor: ReviewPreprocessor = None

def main():
    ver = "0.8.0-experimental"
    print(f"Amazon Cluster Analysis v{ver}")
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Jsonl file to process including extension")
    parser.add_argument("-s", "--seed", type=int, help="Random seed (default 1967)", default=1967)
    parser.add_argument("-m", "--max-reviews", type=int, help="Reviews to process (default 1000)", default=1000)
    parser.add_argument("-v", "--version", action="version", version=f"{ver}")
    parser.add_argument("-r", "--runs", type=int, help="Number of runs (default 1)", default=1)
    parser.add_argument("-n", "--noimages", action="store_true", help="Do not show images, useful for batch processing")
    parser.add_argument("-hc", "--hcluster", type=int, help="Minimum cluster size for HDBSCAN (default 3)", default=3)
    parser.add_argument("-hs", "--hsample", type=int, help="Minimum samples for HDBSCAN (default 2)", default=2)
    parser.add_argument("-t", "--threshold", type=float, help="Relevance threshold for clustering (default 0.25)", default=0.25)
    parser.add_argument("-b", "--bypass", action="store_true", help="Bypass caches")
    parser.add_argument("-e", "--earlystop", action="store_true", help="Stop after clusters have been printed")
    parser.add_argument("-fr", "--forcerandom", action="store_true", help="Use random scores instead of sentiment analysis with LLM")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"File {args.filename} not found.")
        return


    # Data are stored into a directory named after the topic and the seed
    # For example, if seed is 1967 and the topic is "general", the directory will be "general/1967"
    seed = args.seed
    file_path = args.filename
    topicGeneral = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.exists(topicGeneral):
        os.makedirs(topicGeneral)
    topicAndSeed = os.path.join(topicGeneral, str(seed))
    if not os.path.exists(topicAndSeed):
        os.makedirs(topicAndSeed)
    # Set cache files (shared amongst the modules)
    sentiments_cache_file = os.path.join(topicAndSeed, f"{topicGeneral}_sentiments_cache.json")
    embeddings_cache_file = os.path.join(topicGeneral, f"{topicGeneral}_embeddings_cache.pkl")
    preprocessing_cache_prefix = os.path.join(topicGeneral, topicGeneral)
    # Set result file (csv format, containing original and adjusted ratings)
    result_file = os.path.join(topicGeneral, f"{topicGeneral}_results.csv")

    print(f"""This run uses random seed {seed}
Files are stored in the following structure (see README.md for details):

|-- {topicGeneral}
    |-- {topicGeneral}_results.csv
    |-- {topicGeneral}_embeddings_cache.pkl
    |-- {topicGeneral}_preprocessing_cache.json
    |-- {seed}
        |-- {topicGeneral}_sentiments_cache.json
        
"""
    )

    # Initialize embeddings cache
    embeddingManager = EmbeddingsManager(
        embeddings_cache_file=embeddings_cache_file,
        prevent_cache=args.bypass
    )
    preprocessor = ReviewPreprocessor(preprocessing_cache_prefix)

    # Initialize sentiment cache
    sentimentCache_init(sentiments_cache_file, args.bypass)
    # Initialize preprocessing cache

    label_text = "text"
    label_rating = "rating"

    # Load the SpaCy model
    nlp = spacy.load("en_core_web_sm")
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    reviewTopicFile = os.path.join(topicAndSeed, f"{topicGeneral}_topics.json")

    # Starting from this point, the code is repeated for each run
    for run in range(args.runs):

        original_reviews: list[dict] = []
        original_indices: set[int] = set()
        preprocessed_reviews: list[str] = []
        typicalTopics: list[str] = []
        bagOfWords: list[str] = []
        counter: int = 0
        specific_topics: list[str] = []
        generic_topics: list[str] = []
        irrelevantReviews: list[str] = []
        generic_clusters: list[tuple] = []
        specific_clusters: list[tuple] = []
        aggregated_clusters: list[tuple] = []

        pairs: list[tuple[str, str]] = []
        nouns: list[str] = []

        print(f"Run {run + 1}/{args.runs}")
        # Load a random sample of reviews from the file, preprocess them, then
        # calculate the embeddings of the words found in the reviews.
        original_reviews, original_indices = preprocessor.load_reviews(file_path, args.max_reviews, label_text, label_rating,seed)
#        preprocessed_reviews = preprocess_and_extract_topics(original_reviews)

        print("Preprocessing reviews...")
        # Build a dict mapping review text to its rating. This is useful to save the data in the cache.
        reviews_dict: dict[str, float] = {str(review[label_text]): review[label_rating] for review in original_reviews}
        preprocessed_reviews = preprocessor.preprocess_reviews(reviews_dict)        # Updated: use the preprocessor instance

        # Extract adjective-noun pairs
        for review in preprocessed_reviews.keys():
            # Initialize the list for this review
            # Avoid splitting the sentence when the period is part of an abbreviation
            review = review.replace("dr.", "dr").replace("mr.", "mr").replace("prof.", "prof")
            review = review.replace("etc.", "etc").replace("e.g.", "eg").replace("i.e.", "ie")
            reviews_dict[review] = []

            # split sentences on hard punctuation (periods, exclamation marks, question marks)
            sentences = re.split(r'(?<=[.!?]) +', review)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 4:
                    continue
                # Process the sentence with SpaCy
                doc = nlp(sentence)
                pairs = []
                nouns = []
                for token in doc:
                    if token.pos_ == "NOUN":
                        for child in token.children:
                            if child.pos_ == "ADJ":
                                pairs.append((token.text, child.text))
                                nouns.append(token.text)
                nouns = sorted(list(set(nouns)))
                reviews_dict[review].append({"sentence": sentence, "nouns": nouns, "pairs": pairs})




        import json

        listOfTopics: dict[str,str] = {}
        # Loads the topics from the file if it exists
        if os.path.exists(reviewTopicFile):
            with open(reviewTopicFile, "rt", encoding="utf-8") as f:
                json_data = f.read()
                if json_data != "":
                    listOfTopics = json.loads(json_data)                    

        # Invoke the LLM to get the typical topics for the reviews
        print("Invoking LLM to get typical topics...", end="")
        for item in preprocessed_reviews.values():
            sublistofTopics: list[str] = []
            review: str = item["corrected"]
            if review in listOfTopics:
                # Skip reviews already processed
                continue
            # preprocess the review to avoid splitting on forms like dr., mr., prof., etc.
            review = review.replace("dr.", "dr").replace("mr.", "mr").replace("prof.", "prof")
            review = review.replace("etc.", "etc").replace("e.g.", "eg").replace("i.e.", "ie")
            for sentence in review.split("."):
                # Skip empty sentences
                if len(sentence.strip()) < 4:
                    continue
                try:
                    topics: list[str] = sentiment_topicsFromSentence(sentence)
                    if len(topics) == 0:
                        continue
                    # Filter out single-character topics (likely letters)
                    for topic in topics:
                        if (topic is not None) and (len(topic) > 1):
                            sublistofTopics.append(topic)
                except Exception as e:
                    print(f"Error: {e}")
            listOfTopics[review] = sublistofTopics

        # Remove duplicates and empty strings
        listOfTopics = {k: list(set(filter(None, v))) for k, v in listOfTopics.items()} 
        print(" done.")
        # Save the topics to the file
        with open(reviewTopicFile, "wt", encoding="utf-8") as f:
            json.dump(listOfTopics, f, ensure_ascii=False, indent=4)
        f.close()

        bagOfWords = [term for topic in listOfTopics.values() for term in topic]
        # Remove duplicates and empty strings   
        bagOfWords = list(set(filter(None, bagOfWords)))
        # correct and lemmatize using the preprocessor (LLM may return words in different forms)
        # Before invoking the function, we are obliged to create a fake dict, because
        # the function expects a dict as input.
        word_dict: dict[str, float] = {word: 0 for word in bagOfWords}
        bagOfWords = preprocessor.preprocess_reviews(word_dict)
        # The function returns a dict, so we need to extract the keys
        # This is cumbersome, maybe we can write a different function to do this...
        bagOfWords = list(bagOfWords.keys())

        # Remove duplicates and empty strings   
        bagOfWords = list(set(filter(None, bagOfWords)))
        # add topics to embedding cache
        embeddingManager.cacheTopicEmbeddings(bagOfWords)
        # calculate sentence embeddings for the reviews
        print("Calculating sentence embeddings...", end="")

        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Calculate the embeddings for the reviews
        reviewsEmbeddings: dict[str, list[float]] = {}
        corrected_reviews: dict[str, str] = {}
        corrected_reviews = {review: item["corrected"] for review, item in preprocessed_reviews.items()}
        embeddings = emb_model.encode(list(review for review in corrected_reviews), show_progress_bar=True, device="cpu")
        # Store the embeddings in a dictionary, associating each review with its embedding
        # This is not necessary, but it allows to check the embeddings for each review
        reviewsEmbeddings = {review: embedding for review, embedding in zip(corrected_reviews, embeddings)}

        # reviewsEmbeddings now associates the review with its embedding
        # Now for each review embeddings, extract the three most relevant topics,
        # using the semantic similarity.
        topicsPerReview: dict[str, list[str]] = {}
        for idx, (review, embeddings) in enumerate(reviewsEmbeddings.items()):
            # Get the topics for each review
            distances: dict[str, float] = {}
            for word in bagOfWords:
                # Calculate the semantic similarity between the review and the topic
                distances[word] = embeddingManager.getSimilarityDistance(embeddings, embeddingManager.getEmbedding(word))
            # Sort the distances and get the three most relevant topics
            # Get the three most relevant topics    
            relevant_topics = [topic for topic, distance in sorted(distances.items(), key=lambda x: x[1]) if distance <= 0.5][:3]
            # Save the topics in the dictionary
            topicsPerReview[review] = relevant_topics


        bagOfWords = [term for review in preprocessed_reviews for term in review.split()]
        # add specific and general topics to the bag of words
        bagOfWords.extend([topicGeneral])
        typicalTopics = sentiment_getTypicalTopics(topicGeneral)
        bagOfWords.extend(typicalTopics)
        bagOfWords.extend("purchase")
        print(" completed.")
        bagOfWords = list(set(filter(None, bagOfWords)))
        # Cache the embeddings of the words found in the reviews
        embeddingManager.cacheTopicEmbeddings(bagOfWords)




        # Extract the topics semantically related to the general topic of the reviews.
        print(f"Extracting review-specific topics (similarity threshold {args.threshold})")
        specific_topics, _, irrelevantReviews = embeddingManager.extract_relevant_topics(
            reviews=preprocessed_reviews,
            referenceTopic=topicGeneral,
            relevance_threshold=args.threshold,
        )
        counter += len(irrelevantReviews)
        generic_topics, _, irrelevantReviews = embeddingManager.extract_relevant_topics(
            reviews=preprocessed_reviews,
            referenceTopic="purchase",
            relevance_threshold=args.threshold,
        )
        counter += len(irrelevantReviews)
        # Cluster topics based on embeddings calculated during pre-processing
        if len(specific_topics) > 0:
            print(f"Creating relevant clusters (HS={args.hcluster}, HM={args.hsample})...")
            specific_clusters = embeddingManager.clustering_topics(
                relevant_topics=specific_topics,
                cluster_size=args.hcluster,
                min_samples=args.hsample,
    #            min_topics=5
            )

        # Now do the same for the purchase topic, but instead of using all the reviews,
        # we use only the reviews not matching the general topic.
        if 1:
#        for topic in typicalTopics:
            if len(generic_topics) > 0:
                generic_clusters = embeddingManager.clustering_topics(
                    relevant_topics=generic_topics,
                    cluster_size=args.hcluster,
                    min_samples=args.hsample,
#                    min_topics=3
                )
        
        aggregated_clusters = generic_clusters + specific_clusters
        print(f"Irrelevant reviews {counter}/{len(preprocessed_reviews)}.")

#        aggregated_clusters = generic_clusters + specific_clusters
#        specific_topics.extend(generic_topics)


        # Recalculate the centroids. Those obtained by calculating the
        # average of the embeddings of the words in the cluster don't
        # seem very useful.

        # This is a dictionary where the key is the cluster label and the value
        #  is the list of words in the cluster.
        lookupTable: dict[str, str] = {}

        print("Invoking LLM to calculate pseudo-centroids...", end="")
        for label, cluster_info in aggregated_clusters:
            cluster_info['centroid'] = sentiment_returnMostRelevantTopic(
                cluster_info['sample_words']
            )
            # Save the centroid in the lookup table
            lookupTable[label] = cluster_info['centroid']
            print(".", end="", flush=True)
        print(" done.")


        print("\nMost important clusters:")
        print(f"{'Label':<15}{'Centroid':<20}{'Frequency':<15}{'N. aspects':<12}{'Keywords'}")
        print("=" * 150)
        for label, cluster_info in aggregated_clusters:
            print(f"{label:<15}{cluster_info['centroid']:<20}{cluster_info['frequency']:<15}"
                f"{cluster_info['n_aspects']:<12}{cluster_info['sample_words']}")

        if (not args.earlystop):

            # Now we use the list of words in the clusters to find the topics in the reviews
            # to be used for the sentiment analysis.
            # So we check if at least word of a given cluster is in the review, and if so,
            # we associate the review with the related centroid.
            #
            # For example, if a cluster contain { magazine, publication, article } and the
            # centroid is "magazine", then if a review contains the word "publication" we
            # associate the review with the centroid "magazine".
            # This allows to search the sentiments for a limited number of topics,
            # instead of the whole cluster.

            listOfCentroids: list[str] = []
            topicsAndDetails: list[tuple[str, float, list[str]], bool] = []
            for o_review, p_review in zip(original_reviews, preprocessed_reviews):
                listOfCentroids = []
                # For each review, we check if the review contains at least one word of a cluster.
                for _, cluster_info in aggregated_clusters:
                    if any(word in p_review.split() for word in cluster_info["sample_words"].split(",")):
                        # If so, we add the centroid to the list of centroids.
                        listOfCentroids.append(cluster_info["centroid"])
                # There shouldn't be duplicates, but just in case we remove them.
                listOfCentroids = list(set(listOfCentroids))
                # Now we save the original text, its rating and the list of associated centroids.
                topicsAndDetails.append((o_review["text"], o_review["overall"], listOfCentroids, args.forcerandom))

            # Write in the log the associated centroids for each review
            with open(os.path.join(topicAndSeed, f"{topicGeneral}_clusters.txt"), "wt", encoding="utf-8") as f:
                for o_review, p_review, listOfCentroids, _ in topicsAndDetails:
                    f.write(f"{o_review}\n{p_review}\n{listOfCentroids}\n\n")
                f.close()    

            # count the number of reviews not belonging to any cluster
            unclustered_count = sum(1 for detail in topicsAndDetails if len(detail[2]) == 0)
            print(f"\n{len(topicsAndDetails)} reviews of which {unclustered_count} orphan.")

            # Not executed
            if (0 ):
                print("Aggregating similar topics...", end="", flush=True)
                # For each cluster, aggregate similar topics to simplify the work of the LLM
                for _, cluster_info in aggregated_clusters:
                    cluster_info['sample_words'] = sentiment_aggregateSimilarTopics(cluster_info['sample_words'])
                    cluster_info['n_aspects'] = len(cluster_info['sample_words'])
                    print(".", end="", flush=True)
                print(" done.")

            print("Updating sentiments (. = calculated, _ = skipped, C = cached, E = error, J Json error)\n")
            calculatedSentiments: list[tuple[str, float]] = []
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=6) as executor:
                calculatedSentiments = list(executor.map(lambda t: sentimentCache_getSentimentAndAdjustedRating(t[0], t[1], t[2], t[3]), topicsAndDetails))

#            original_ratings: np.ndarray = np.array([t[1] for t in topicsAndDetails if t[2] != []])
#            adjusted_ratings: np.ndarray = np.array([result[1] for result in calculatedSentiments if result[0] != {}])
            original_ratings: np.ndarray = np.array([t[1] for t in topicsAndDetails])
            adjusted_ratings: np.ndarray = np.zeros(len(original_ratings))
            for idx, result in enumerate(calculatedSentiments):
                score: float = 0.0
                for value in result[0]:
                    score += result[0][value]
            adjusted_ratings[idx] = score
            # This operation allows to change the strength of the adjustment
            adjustements: np.ndarray = (adjusted_ratings - original_ratings)
            adjusted_ratings = original_ratings + adjustements * 0.5
            adjusted_ratings = adjustements * 0.5
            print("\nDone.")

            # Store results in a json file
            if (not os.path.exists(result_file)):
                with open(result_file, "wt", encoding="utf-8") as f:
                    f.write("timestamp,run,original,adjusted,forcerandom,seed,reviewID,sentence,appVersion\n")
                    f.close()
            with open(result_file, "at", encoding="utf-8") as f:
                for idx in range(len(calculatedSentiments)):
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{run+1},"
                            f"{original_ratings[idx]},{adjusted_ratings[idx]},"
                            f"{int(args.forcerandom)},"
                            f"{seed},{original_indices.pop()},"
                            f"\"{topicsAndDetails[idx][0][:48].replace('\n','')}...\",{ver}\n")
                f.close()

            # If args.noimages is not set, plot the results
            if (not args.noimages):

                # Aggregate and plot results

                score_range = np.arange(-12.0, 12.0, 0.5)
                adjusted_counts = {round(r,1): 0 for r in np.round(score_range,1)}
                for rating in adjusted_ratings:
                    rounded_rating = round(rating*2)/2
                    if rounded_rating in adjusted_counts:
                        adjusted_counts[rounded_rating] += 1

                plt.figure(figsize=(16, 9))
                plt.subplot(2, 1, 1)
                bins = np.arange(0.5, 6.5, 1)
                plt.hist(original_ratings, bins=bins, color="blue", alpha=0.7, edgecolor="black")
                plt.xlabel("Original Rating")
                plt.ylabel("Count")
                plt.title("Distribution of Original Ratings")
                plt.xticks(np.arange(1, 6, 1))
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                plt.subplot(2, 1, 2)
                plt.bar(list(adjusted_counts.keys()), list(adjusted_counts.values()),
                        color="green", alpha=0.7, edgecolor="black", width=0.1)
                plt.xlabel("Adjusted Rating")
                plt.ylabel("Count")
                plt.title("Distribution of Adjusted Ratings")
                plt.grid(axis="both", linestyle="solid", alpha=0.7)
                plt.minorticks_on()
                plt.show()
                input("Press Enter to continue...")
            else:
                print("Images not shown. Use -n to show them.")
        else:
            print("Early stop requested. No sentiment analysis performed.") 

if __name__ == "__main__":
    main()
