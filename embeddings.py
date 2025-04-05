import os
import pickle
import warnings
import numpy as np
import hdbscan
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from collections import Counter

warnings.simplefilter(action='ignore')

# Global variables for embeddings and the overall topic
topic_embeddings_global = {}
overallTopicEmbedding = None
# variable to prevent embedding caching
preventEmbeddingCaching: bool = False
# Global variable for the embeddings cache file path
_EMBEDDINGS_CACHE_FILE = ""

def embeddingsCache_init(embeddings_cache_file: str, prevent_cache: bool = False):
    """
    Initialize and load the embeddings cache variables. If prevent_cache is set
    to True, the cache remains empty and is not saved, but can still be used
    """
    global _EMBEDDINGS_CACHE_FILE, topic_embeddings_global
    global overallTopicEmbedding, preventEmbeddingCaching
    _EMBEDDINGS_CACHE_FILE = embeddings_cache_file
    preventEmbeddingCaching = prevent_cache
    topic_embeddings_global = {}
    overallTopicEmbedding = None
    embeddingCache_load()

def embeddingCache_load():
    global _EMBEDDINGS_CACHE_FILE, topic_embeddings_global, preventEmbeddingCaching
    # Load the embeddings from the cache file unless preventEmbeddingCaching is set to True
    if (not preventEmbeddingCaching) and os.path.exists(_EMBEDDINGS_CACHE_FILE):
        with open(_EMBEDDINGS_CACHE_FILE, "rb") as f:
            topic_embeddings_global = pickle.load(f)

def embeddingCache_save():
    global _EMBEDDINGS_CACHE_FILE, topic_embeddings_global, preventEmbeddingCaching
    # Save the embeddings to the cache file unless preventEmbeddingCaching is set to True
    if (not preventEmbeddingCaching):
        with open(_EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(topic_embeddings_global, f)

def is_relevant_topic(topic: str, threshold: float = 0.25) -> bool:
    global overallTopicEmbedding, topic_embeddings_global
    topic_emb = topic_embeddings_global.get(topic)
    if topic_emb is None:
        return False
    if np.linalg.norm(overallTopicEmbedding) == 0 or np.linalg.norm(topic_emb) == 0:
        return False
    similarity = np.dot(overallTopicEmbedding, topic_emb) / (np.linalg.norm(overallTopicEmbedding) * np.linalg.norm(topic_emb))
    return similarity >= threshold

def clustering_topics(
        reviews: list[str],
        relevance_threshold: float = 0.25,
        cluster_size: int = 3,
        min_samples: int = 2,
    ) -> list[tuple]:
    """
    Clusters the topics based on their embeddings and returns the most important clusters.
    
    Args:
        reviews: List of reviews.
        relevance_threshold: minimum semantic similarity for topics to be considered relevant.
        cluster_size: minimum size of clusters for HDBSCAN.
        min_samples: minimum number of samples in a cluster for HDBSCAN.
    Returns:
        list[tuple]: List of tuples containing cluster information.
    """
    global topic_embeddings_global, overallTopicEmbedding
    print(f"Creating relevant clusters (ST={relevance_threshold}, HS={cluster_size}, HM={min_samples})...")
    print("One dot = 1000 reviews", end="", flush=True)
    relevant_topics = []
    for i, review in enumerate(reviews):
        # Create a list of relevant words from each review. To decide if a word is relevant,
        # we check its semiamntic similarity with the overall topic embedding.
        # Note that is_relevant_topic() also updates the global topic_embeddings_global dictionary.
        relevant_topics.extend([t for t in review.split() if is_relevant_topic(t,relevance_threshold)])
        if i and i % 1000 == 0:
            print(".", end="", flush=True)
    print(" done.")

    topic_counts = Counter(relevant_topics)
    topics_list = list(topic_counts.keys())
    embeddings = [topic_embeddings_global[topic] for topic in topics_list]
#    embeddings_array = np.array(embeddings)

    # Apply PCA to reduce dimensionality
    # We use 0.9 to keep 90% of the variance in the data
    pca = PCA(n_components=0.9)
#    reduced_vectors = pca.fit_transform(embeddings_array)
    reduced_vectors = pca.fit_transform(embeddings)
    # Transform the overall topic embedding to the reduced space
    # This is needed to calculate the distance of the overall topic embedding from the clusters
    overallTopicEmbedding = pca.transform(np.array([overallTopicEmbedding]))
    del pca

    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(reduced_vectors)
    
    centroids = {label: np.mean(reduced_vectors[labels == label], axis=0) for label in set(labels) if label != -1}
    clusters = {label: [topics_list[i] for i in range(len(labels)) if labels[i] == label]
                for label in set(labels) if label != -1}
    topic_embeddings = {topic: emb for topic, emb in zip(topics_list, reduced_vectors)}

    most_important_clusters = get_most_important_clusters(clusters, topic_counts, centroids, topic_embeddings, n=10)
    return most_important_clusters

def find_nearest_topic(centroid, topics, topic_embeddings) -> str:
    distances = cdist([centroid], np.array([topic_embeddings[t] for t in topics]), metric='euclidean')
    return topics[np.argmin(distances)]

def get_most_important_clusters(clusters, topic_counts, centroids, topic_embeddings, n) -> list[tuple]:
    cluster_frequencies = {}
    for label, topics in clusters.items():
        total_frequency = sum(topic_counts[t] for t in topics)
        centroid_word = find_nearest_topic(centroids[label], topics, topic_embeddings)
        sample_words = ", ".join(topics)
        cluster_frequencies[label] = {
            "centroid": centroid_word,
            "sample_words": sample_words,
            "n_aspects": len(topics),
            "frequency": total_frequency
        }
    sorted_clusters = sorted(cluster_frequencies.items(), key=lambda x: x[1]["frequency"], reverse=True)
    return sorted_clusters[:n]

def describe_general_topic(topic: str) -> str:
    """
    Describe the general topic using an LLM model.
    :param topic: the general topic to describe
    :return: the description of the topic
    """
    from mistralai import Mistral
    from key import MistraAIKey as api_key

    response: str = ""
    prompt: str = f"""Read the following topic:
            {topic}
            ---
            Describe the topic in detail, including its meaning and context.
            Be specific and informative, but also concise.
            """
    # Initialize the Mistral client
    genAI_Client: Mistral = Mistral(api_key=api_key)
    try:
        response = genAI_Client.chat.complete(
            model="mistral-small-latest",
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        response = topic  # Fallback to the original topic if there's an error
    # Return the response from the LLM model
    return topic

def process_topic_extraction(preprocessed_reviews: list[str], topic_general: str):
    """
    """
    global topic_embeddings_global, overallTopicEmbedding

    emb_model: SentenceTransformer = None
    extracted: list[str] = []
    new_count: int = 0
    descriptiveTopic: str = ""

    # Load the transformer model and calculate the overall topic embedding
    # To do so, instead of using just the database name, we invoke an LLM
    # to get a more detailed description of the topic.
    print("Importing transformer library...")
    from sentence_transformers import SentenceTransformer
    print("Loading transformer model...")
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Ask the LLM to describe the general topic
    descriptiveTopic = describe_general_topic(topic_general)
    # then calculate the embedding
    overallTopicEmbedding = emb_model.encode(descriptiveTopic)
#    extract_and_cache_embeddings(preprocessed_reviews, emb_model)

    print("Extracting topics...", end="", flush=True)
    for review in preprocessed_reviews:
        extracted.extend(review.split())
    extracted = sorted(set(extracted))
    print(f"{len(extracted)} unique topics")
    print("Calculating embeddings...", end="", flush=True)

    embeddingCache_load()
    new_count = 0
    for i, topic in enumerate(extracted):
        if topic not in topic_embeddings_global:
            topic_embeddings_global[topic] = emb_model.encode(topic)
            new_count += 1
        if i and i % 1000 == 0:
            print(".", end="", flush=True)
            embeddingCache_save()
    print(f" done (new embeddings: {new_count} over {len(extracted)})")
    embeddingCache_save()

#    del emb_model
