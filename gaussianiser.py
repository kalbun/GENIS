import os
import random
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import load_reviews, preprocess_and_extract_topics
from embeddings import (
    embeddingsCache_init,
    clustering_topics,
    process_topic_extraction
)
from sentiments import (
    sentimentCache_init,
    sentimentCache_getSentimentAndAdjustedRating
)

def main():
    ver = "0.4.0"
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
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"File {args.filename} not found.")
        return


    # Data are stored into a directory named after the topic and the seed
    # For example, if seed is 1967 and the topic is "general", the directory will be "general/1967"
    seed = args.seed
    random.seed(seed)
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
    # Set result file (csv format, containing original and adjusted ratings)
    result_file = os.path.join(topicGeneral, f"{topicGeneral}_results.csv")

    print(f"""This run uses random seed {seed}
Files are stored in the following structure:
.
|-- {topicGeneral}
    |-- {topicGeneral}_results.csv
    |-- {topicGeneral}_sentiments_cache.json
    |-- {seed}
        |-- {topicGeneral}_embeddings_cache.pkl
"""
    )

    # Initialize embeddings cache
    embeddingsCache_init(embeddings_cache_file)
    # Initialize sentiment cache
    sentimentCache_init(sentiments_cache_file, args.bypass)

    label_text = "text"
    label_rating = "rating"

    # Starting from this point, the code is repeated for each run
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")

        original_reviews: list[dict] = []
        original_indices: set[int] = set()
        preprocessed_reviews: list[str] = []

        # Load and preprocess reviews
        original_reviews, original_indices = load_reviews(file_path, args.max_reviews, label_text, label_rating)
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

        print("\nMost important clusters:")
        print(f"{'Label':<15}{'Centroid':<20}{'Frequency':<15}{'N. aspects':<12}{'Keywords'}")
        print("=" * 150)
        for label, cluster_info in most_important_clusters:
            print(f"{label:<15}{cluster_info['centroid']:<20}{cluster_info['frequency']:<15}"
                f"{cluster_info['n_aspects']:<12}{cluster_info['sample_words']}")

        if (not args.earlystop):
            # Process each review for sentiment update
            print("Updating sentiments (. = calculated, C = cached, E = error, J Json error)\n")
            topicsAndDetails: list[tuple[str, float, list[str]]] = []
            for review in original_reviews:
                topics = [topic for _, cluster_info in most_important_clusters
                        for topic in cluster_info["sample_words"].split(", ")
                        if topic.lower() in review["text"].lower()]
                topicsAndDetails.append((review["text"], review["overall"], topics))

            calculatedSentiments: list[tuple[str, float]] = []
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=6) as executor:
                calculatedSentiments = list(executor.map(lambda t: sentimentCache_getSentimentAndAdjustedRating(t[0], t[1], t[2]), topicsAndDetails))

            original_ratings: np.ndarray = np.array([t[1] for t in topicsAndDetails])
            adjusted_ratings: np.ndarray = np.array([result[1] for result in calculatedSentiments])
            # This operation allows to change the strength of the adjustment
            adjustements: np.ndarray = (adjusted_ratings - original_ratings)
            adjusted_ratings = original_ratings + adjustements * 0.5
            print("\nDone.")

            # Store results in a json file
            if (not os.path.exists(result_file)):
                with open(result_file, "wt", encoding="utf-8") as f:
                    f.write("timestamp,run,original,adjusted,seed,reviewID,sentence,appVersion\n")
                    f.close()
            with open(result_file, "at", encoding="utf-8") as f:
                for idx in range(len(calculatedSentiments)):
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{run+1},"
                            f"{original_ratings[idx]},{adjusted_ratings[idx]},"
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
                plt.hist(original_ratings, bins=5, color="blue", alpha=0.7, edgecolor="black")
                plt.xlabel("Original Rating")
                plt.ylabel("Count")
                plt.title("Distribution of Original Ratings")
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                plt.subplot(2, 1, 2)
                plt.bar(list(adjusted_counts.keys()), list(adjusted_counts.values()),
                        color="green", alpha=0.7, edgecolor="black", width=0.1)
                plt.xlabel("Adjusted Rating")
                plt.ylabel("Count")
                plt.title("Distribution of Adjusted Ratings")
                plt.grid(axis="both", linestyle="--", alpha=0.7)
                plt.show()

if __name__ == "__main__":
    main()
