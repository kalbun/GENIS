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
_EMBEDDINGS_CACHE_FILE = ""

def initialise(topic_general: str, embeddings_cache_file: str):
    global _EMBEDDINGS_CACHE_FILE, overallTopicEmbedding, topic_embeddings_global, genAI_Client
    _EMBEDDINGS_CACHE_FILE = embeddings_cache_file

def extract_and_cache_embeddings(preprocessed_reviews: list[str], emb_model):
    global topic_embeddings_global, _EMBEDDINGS_CACHE_FILE
    extracted = []
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

def embeddingCache_load():
    global _EMBEDDINGS_CACHE_FILE, topic_embeddings_global
    if os.path.exists(_EMBEDDINGS_CACHE_FILE):
        with open(_EMBEDDINGS_CACHE_FILE, "rb") as f:
            topic_embeddings_global = pickle.load(f)

def embeddingCache_save():
    global _EMBEDDINGS_CACHE_FILE, topic_embeddings_global
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

def apply_pca_to_embeddings(embeddings, n_components: float) -> tuple[np.array, PCA]:
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings), pca

def clustering_topics(preprocessed_reviews: list[str]) -> list[tuple]:
    global topic_embeddings_global, overallTopicEmbedding
    print("Creating relevant clusters...", end="", flush=True)
    relevant_topics = []
    for i, review in enumerate(preprocessed_reviews):
        relevant_topics.extend([t for t in review.split() if is_relevant_topic(t)])
        if i and i % 1000 == 0:
            print(".", end="", flush=True)
    print(" done.")

    topic_counts = Counter(relevant_topics)
    topics_list = list(topic_counts.keys())
    embeddings = [topic_embeddings_global[topic] for topic in topics_list]
    embeddings_array = np.array(embeddings)

    reduced_vectors, pca = apply_pca_to_embeddings(embeddings_array, n_components=0.90)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    labels = clusterer.fit_predict(reduced_vectors)
    
    centroids = {label: np.mean(reduced_vectors[labels == label], axis=0) for label in set(labels) if label != -1}
    clusters = {label: [topics_list[i] for i in range(len(labels)) if labels[i] == label]
                for label in set(labels) if label != -1}
    topic_embeddings = {topic: emb for topic, emb in zip(topics_list, embeddings)}

    most_important_clusters = get_most_important_clusters(clusters, topic_counts, centroids, topic_embeddings, pca, n=10)
    return most_important_clusters

def find_nearest_topic(centroid, topics, topic_embeddings, pca) -> str:
    transformed = pca.transform(np.array([topic_embeddings[t] for t in topics]))
    distances = cdist([centroid], transformed, metric='euclidean')
    return topics[np.argmin(distances)]

def get_most_important_clusters(clusters, topic_counts, centroids, topic_embeddings, pca, n) -> list[tuple]:
    cluster_frequencies = {}
    for label, topics in clusters.items():
        total_frequency = sum(topic_counts[t] for t in topics)
        centroid_word = find_nearest_topic(centroids[label], topics, topic_embeddings, pca)
        sample_words = ", ".join(topics)
        cluster_frequencies[label] = {
            "centroid": centroid_word,
            "sample_words": sample_words,
            "n_aspects": len(topics),
            "frequency": total_frequency
        }
    sorted_clusters = sorted(cluster_frequencies.items(), key=lambda x: x[1]["frequency"], reverse=True)
    return sorted_clusters[:n]

def process_topic_extraction(preprocessed_reviews: list[str], topic_general: str):
    print("Importing transformer model...")
    from sentence_transformers import SentenceTransformer
    print("Loading transformer model...")
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    global overallTopicEmbedding
    overallTopicEmbedding = emb_model.encode(topic_general)
    extract_and_cache_embeddings(preprocessed_reviews, emb_model)
#    del emb_model
