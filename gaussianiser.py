import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import (
    load_reviews, preprocess_and_extract_topics,
    preprocessingCache_init
)
from embeddings import (
    embeddingsCache_init,
    clustering_topics,
    process_topic_extraction
)
from sentiments import (
    sentimentCache_init,
    sentimentCache_getSentimentAndAdjustedRating,
    sentiment_aggregateSimilarTopics,
    sentiment_returnMostRelevantTopic
)

def main():
    ver = "0.7.0-experimental"
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
    preprocessing_cache_file = os.path.join(topicGeneral, f"{topicGeneral}_preprocessing_cache.json")
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
    embeddingsCache_init(embeddings_cache_file)
    # Initialize sentiment cache
    sentimentCache_init(sentiments_cache_file, args.bypass)
    # Initialize preprocessing cache
    preprocessingCache_init(preprocessing_cache_file)

    label_text = "text"
    label_rating = "rating"

    # Starting from this point, the code is repeated for each run
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")

        original_reviews: list[dict] = []
        original_indices: set[int] = set()
        preprocessed_reviews: list[str] = []

        # Load and preprocess reviews
        original_reviews, original_indices = load_reviews(file_path, args.max_reviews, label_text, label_rating,seed)
        preprocessed_reviews = preprocess_and_extract_topics(original_reviews)

        process_topic_extraction(preprocessed_reviews,topic_general=topicGeneral)

        # Cluster topics based on embeddings calculated during pre-processing
        most_important_clusters: list[tuple] = []
        most_important_clusters = clustering_topics(
            reviews=preprocessed_reviews,
            relevance_threshold=args.threshold,
            cluster_size=args.hcluster,
            min_samples=args.hsample,
        )

        # Recalculate the centroids. They are obtained by calculating the
        # average of the embeddings of the words in the cluster, but this
        # doesn't seem to work well. So we use an LLM instead.
        for idx, (label, cluster_info) in enumerate(most_important_clusters):
            cluster_info['centroid'] = sentiment_returnMostRelevantTopic(cluster_info['sample_words'])

        print("\nMost important clusters:")
        print(f"{'Label':<15}{'Centroid':<20}{'Frequency':<15}{'N. aspects':<12}{'Keywords'}")
        print("=" * 150)
        for label, cluster_info in most_important_clusters:
            print(f"{label:<15}{cluster_info['centroid']:<20}{cluster_info['frequency']:<15}"
                f"{cluster_info['n_aspects']:<12}{cluster_info['sample_words']}")

        if (not args.earlystop):

            # This is the main part of the code where the sentiment analysis is performed.
            # The operations are as follows:
            # 1. create a list where each review is associated with a numnber of topics
            #    extracted from the most important clusters. There is no one-to-one correspondence
            #    between reviews and topics, but this is ok.
            #    The review and its associated topics are stored in a list of tuples
            # 2. aggregate similar topics stored in most_important_clusters. The reason to do
            #    it after the association and not before is that we don't know which topics of
            #    a certain cluster are found in the reviews. Aggregating them before would
            #    possibly eliminate some topics found in the reviews.
            # 3. check again the topics associated to each review and remove those no longer
            #    found. The idea is the that passing just the aggregated topics to the LLM
            #    will not prevent it from finding other topics.

            # Check to which topic(s) each review belongs to
            # This is done by checking if the topic is in the review text.
            # If a topic is found, then the centroid of the cluster is used as the topic
            # and the review is associated with that topic.
            # Note that we use the preprocessed reviews and not the original ones! This
            # is needed becauuse otherwise the exact comparison between the topic and the
            # terms in the review would not work.
            # Note that a review can belong to multiple topics.
            listOfCentroids: list[str] = []
            topicsAndDetails: list[tuple[str, float, list[str]], bool] = []
            for o_review, p_review in zip(original_reviews, preprocessed_reviews):
                for _, cluster_info in most_important_clusters:
                    # Check if the topic is in the review text. In this case,
                    # add the centroid to the list and move to the next cluster.
                    if any(word in p_review.split() for word in cluster_info["sample_words"].split(",")):
                        listOfCentroids.append(cluster_info["centroid"])
                # Here we should have a list of centroids for each topic found in the review.
                # We can then replace the original list of topics with the list of centroids.
                topicsAndDetails.append((o_review["text"], o_review["overall"], listOfCentroids, args.forcerandom))
                # Reset the list of centroids for the next review
                listOfCentroids = []

            # count the number of reviews not belonging to any cluster
            unclustered_count = sum(1 for detail in topicsAndDetails if len(detail[2]) == 0)
            print(f"\n{unclustered_count} reviews do not belong to any cluster.")

            if (0 ):
                print("Aggregating similar topics...", end="", flush=True)
                # For each cluster, aggregate similar topics to simplify the work of the LLM
                for _, cluster_info in most_important_clusters:
                    cluster_info['sample_words'] = sentiment_aggregateSimilarTopics(cluster_info['sample_words'])
                    cluster_info['n_aspects'] = len(cluster_info['sample_words'])
                    print(".", end="", flush=True)
                print(" done.")

            # Now remove topics no longer found in the reviews
            all_topics = [topic for _, cluster_info in most_important_clusters for topic in cluster_info['sample_words']]

#            for idx, detail in enumerate(topicsAndDetails):
#                topics = [topic for topic in detail[2] if topic in all_topics]
#                topicsAndDetails[idx] = (detail[0], detail[1], topics, detail[3])

            # Invoke LLM to aggregate similar topics
            print("Updating sentiments (. = calculated, _ = skipped, C = cached, E = error, J Json error)\n")
            calculatedSentiments: list[tuple[str, float]] = []
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=6) as executor:
                calculatedSentiments = list(executor.map(lambda t: sentimentCache_getSentimentAndAdjustedRating(t[0], t[1], t[2], t[3]), topicsAndDetails))

            original_ratings: np.ndarray = np.array([t[1] for t in topicsAndDetails])
            adjusted_ratings: np.ndarray = np.array([result[1] for result in calculatedSentiments])
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
