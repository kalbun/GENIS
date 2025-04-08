import os
import pickle
import warnings
import numpy as np
import hdbscan
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from collections import Counter

warnings.simplefilter(action='ignore')


class EmbeddingsManager:
    def __init__(self, embeddings_cache_file: str, prevent_cache: bool = False):
        """
        Initialize the embeddings manager. If prevent_cache is True, the cache is not saved.
        """
        self.embeddingsCache = {}
        self.overallTopicEmbedding = None
        self.preventEmbeddingCaching: bool = prevent_cache
        self._embeddingsCacheFile = embeddings_cache_file
        self._load_cache()

    def _load_cache(self):
        """Load the embeddings from the cache file unless caching is prevented."""
        if (not self.preventEmbeddingCaching) and os.path.exists(self._embeddingsCacheFile):
            with open(self._embeddingsCacheFile, "rb") as f:
                self.embeddingsCache = pickle.load(f)

    def _save_cache(self):
        """Save the embeddings to the cache file unless caching is prevented."""
        if not self.preventEmbeddingCaching:
            with open(self._embeddingsCacheFile, "wb") as f:
                pickle.dump(self.embeddingsCache, f)

    def is_relevant_topic(self, topic: str, threshold: float = 0.25) -> bool:
        topic_emb = self.embeddingsCache.get(topic)
        if topic_emb is None:
            return False
        if np.linalg.norm(self.overallTopicEmbedding) == 0 or np.linalg.norm(topic_emb) == 0:
            return False
        similarity = np.dot(self.overallTopicEmbedding, topic_emb) / (
            np.linalg.norm(self.overallTopicEmbedding) * np.linalg.norm(topic_emb)
        )
        return similarity >= threshold

    def clustering_topics(
        self,
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
        relevant_topics: list[str] = []
        irrelevantReviewsCount: int = 0

        print(f"Creating relevant clusters (ST={relevance_threshold}, HS={cluster_size}, HM={min_samples})...")
        print("One dot = 1000 reviews", end="", flush=True)

        for i, review in enumerate(reviews):
            current_topics = [t for t in review.split() if self.is_relevant_topic(t, relevance_threshold)]
            if len(current_topics) == 0:
                irrelevantReviewsCount += 1
            else:
                relevant_topics.extend(current_topics)
            if i and i % 1000 == 0:
                print(".", end="", flush=True)
        print(f"\n{len(reviews)} reviews of which {irrelevantReviewsCount} irrelevant.")

        topic_counts = Counter(relevant_topics)
        topics_list = list(topic_counts.keys())
        embeddings = [self.embeddingsCache[topic] for topic in topics_list]

        # Apply PCA to reduce dimensionality (keeping 90% of the variance)
        pca = PCA(n_components=0.9)
        reduced_vectors = pca.fit_transform(embeddings)
        self.overallTopicEmbedding = pca.transform(np.array([self.overallTopicEmbedding]))
        del pca

        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples, metric='euclidean')
        labels = clusterer.fit_predict(reduced_vectors)

        centroids = {label: np.mean(reduced_vectors[labels == label], axis=0)
                     for label in set(labels) if label != -1}
        clusters = {label: [topics_list[i] for i in range(len(labels)) if labels[i] == label]
                    for label in set(labels) if label != -1}
        topic_embeddings = {topic: emb for topic, emb in zip(topics_list, reduced_vectors)}

        most_important_clusters = self.get_most_important_clusters(clusters, topic_counts, centroids, topic_embeddings, n=10)
        return most_important_clusters

    @staticmethod
    def find_nearest_topic(centroid, topics, topic_embeddings) -> str:
        distances = cdist([centroid], np.array([topic_embeddings[t] for t in topics]), metric='euclidean')
        return topics[np.argmin(distances)]

    def get_most_important_clusters(self, clusters, topic_counts, centroids, topic_embeddings, n) -> list[tuple]:
        cluster_frequencies = {}
        for label, topics in clusters.items():
            total_frequency = sum(topic_counts[t] for t in topics)
            centroid_word = self.find_nearest_topic(centroids[label], topics, topic_embeddings)
            sample_words = ", ".join(topics)
            cluster_frequencies[label] = {
                "centroid": centroid_word,
                "sample_words": sample_words,
                "n_aspects": len(topics),
                "frequency": total_frequency
            }
        sorted_clusters = sorted(cluster_frequencies.items(), key=lambda x: x[1]["frequency"], reverse=True)
        return sorted_clusters[:n]

    def describe_general_topic(self, topic: str) -> str:
        """
        Describe the general topic using an LLM model.
        :param topic: the general topic to describe
        :return: the description of the topic
        """
        from mistralai import Mistral
        from key import MistraAIKey as api_key

        prompt = f"""Read the following topic:
        {topic}
        ---
        Describe the topic in detail, including its meaning and context.
        Be specific and informative, but also concise.
        """
        genAI_Client = Mistral(api_key=api_key)
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
        return response

    def process_topic_extraction(self, preprocessed_reviews: list[str], topic_general: str):
        """
        Process topic extraction by calculating embeddings and caching them.
        """
        extracted: list[str] = []
        new_count: int = 0

        print("Extracting topics...", end="", flush=True)
        for review in preprocessed_reviews:
            extracted.extend(review.split())
        extracted = sorted(set(extracted))
        print(f"{len(extracted)} unique topics")

        # Reload cache in case it changed externally
        self._load_cache()
        new_count = 0
        for topic in extracted:
            if topic not in self.embeddingsCache:
                new_count += 1
        if topic_general not in self.embeddingsCache:
            new_count += 1
        else:
            self.overallTopicEmbedding = self.embeddingsCache[topic_general]

        if new_count == 0:
            print("All embeddings already cached")
            return

        # Load the transformer model and calculate the overall topic embedding
        print("Importing transformer library...")
        from sentence_transformers import SentenceTransformer
        print("Loading transformer model...")
        emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Calculating embeddings...", end="", flush=True)
        self.overallTopicEmbedding = emb_model.encode(topic_general)
        self.embeddingsCache[topic_general] = self.overallTopicEmbedding

        new_count = 0
        for i, topic in enumerate(extracted):
            if topic not in self.embeddingsCache:
                self.embeddingsCache[topic] = emb_model.encode(topic)
                new_count += 1
            if i and i % 1000 == 0:
                print(".", end="", flush=True)
                self._save_cache()
        print(f" done (new embeddings: {new_count} over {len(extracted)})")
        self._save_cache()