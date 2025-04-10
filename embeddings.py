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

    def is_relevant_topic(self, topic: str, referenceTopic: str, threshold: float = 0.25) -> bool:
        """
        Check if the topic is relevant to the reference topic based on their embeddings.
        Args:
            topic: str - the topic to check.
            referenceTopic: str - the reference topic.
            threshold: float - the threshold for relevance.
        Returns:
            bool: True if the topic is relevant, False otherwise.
        """
        topic_emb = self.embeddingsCache.get(topic)
        topic_ref__emb = self.embeddingsCache.get(referenceTopic)
        if topic_emb is None:
            return False
        if np.linalg.norm(topic_ref__emb) == 0 or np.linalg.norm(topic_emb) == 0:
            return False
        similarity = np.dot(topic_ref__emb, topic_emb) / (
            np.linalg.norm(topic_ref__emb) * np.linalg.norm(topic_emb)
        )
        return similarity >= threshold

    def extract_relevant_topics(
        self,
        reviews: list[str],
        referenceTopic: str,
        relevance_threshold: float = 0.25,
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Extracts the relevant topics from the reviews based on their embeddings.

        Args:
            reviews: List of reviews.
            referenceTopic: Reference topic to compare against.
            relevance_threshold: minimum semantic similarity for topics to be
                considered relevant.
        Returns:
            list[str]: List of unique and sorted relevant topics.
            list[str]: List of relevant reviews.
            list[str]: List of irrelevant reviews.
        """
        relevant_topics: list[str] = []
        relevantReviews: list[str] = []
        irrelevantReviews: list[str] = []

        for i, review in enumerate(reviews):
            current_topics = [t for t in review.split() if self.is_relevant_topic(t, referenceTopic, relevance_threshold)]
            if len(current_topics) == 0:
                irrelevantReviews.append(review)
            else:
                relevantReviews.append(review)
                relevant_topics.extend(current_topics)
            if i and i % 1000 == 0:
                print(".", end="", flush=True)
        return sorted(set(relevant_topics)), relevantReviews, irrelevantReviews

    def clustering_topics(
        self,
        relevant_topics: list[str],
        cluster_size: int = 3,
        min_samples: int = 2,
        min_topics: int = 4,
    ) -> list[tuple]:
        """
        Clusters the topics based on their embeddings and returns the most important clusters.

        Args:
            relevant_topics: List of relevant topics.
            cluster_size: minimum size of clusters for HDBSCAN.
            min_samples: minimum number of samples in a cluster for HDBSCAN.
            min_topics: minimum number of topics in a cluster to be considered important.
        Returns:
            list[tuple]: List of tuples containing cluster information.
        """

        topic_counts = Counter(relevant_topics)
        topics_list = list(topic_counts.keys())
        embeddings = [self.embeddingsCache[topic] for topic in topics_list]

        # Apply PCA to reduce dimensionality (keeping 90% of the variance)
        pca = PCA(n_components=0.9)
        reduced_vectors = pca.fit_transform(embeddings)
#        self.overallTopicEmbedding = pca.transform(np.array([self.overallTopicEmbedding]))
        del pca

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cluster_size, 
            min_samples=min_samples, 
            metric='euclidean'
        )
        labels = clusterer.fit_predict(reduced_vectors)

        centroids = {label: np.mean(reduced_vectors[labels == label], axis=0)
                     for label in set(labels) if label != -1}
        clusters = {label: [topics_list[i] for i in range(len(labels)) if labels[i] == label]
                    for label in set(labels) if label != -1}
        topic_embeddings = {topic: emb for topic, emb in zip(topics_list, reduced_vectors)}

        most_important_clusters = self.get_most_important_clusters(
            clusters,
            topic_counts, 
            centroids, 
            topic_embeddings, 
            minTopics=min_topics
        )
        return most_important_clusters

    @staticmethod
    def find_nearest_topic(centroid, topics, topic_embeddings) -> str:
        distances = cdist([centroid], np.array([topic_embeddings[t] for t in topics]), metric='euclidean')
        return topics[np.argmin(distances)]

    def get_most_important_clusters(
            self,
            clusters: dict[int, list[str]], 
            topic_counts: dict[str, int],
            centroids: dict[int, np.ndarray],
            topic_embeddings: dict[str, np.ndarray],
            minTopics: int = 7
        ) -> list[tuple]:
        """
        Sort the clusters by their frequency and return the most important ones.
        :param clusters: the clusters to sort
        :param topic_counts: the counts of each topic
        :param centroids: the centroids of each cluster
        :param topic_embeddings: the embeddings of each topic
        :param n: minimum number of topics in a cluster to be considered important
        :return: the most important clusters
        """
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
        # Sort clusters by frequency and return those containing at least n topics
        sorted_clusters = sorted(cluster_frequencies.items(), key=lambda x: x[1]["frequency"], reverse=True)
        return [cluster for cluster in sorted_clusters if cluster[1]['frequency'] > minTopics]

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

    def cacheTopicEmbeddings(self, bagOfWords: list[str]) -> None:
        """
        Calculate the topic embeddings for the given terms and put them in the cache.
        For each topic, the function checks if the embedding is already cached and
        calculates it if not.

        :param bagOfWords: List of terms to calculate the embeddings for.
        :return: None
        """
        new_count: int = 0

        bagOfWords = list(set(bagOfWords))
        print(f"{len(bagOfWords)} unique topics")

        # Reload cache in case it changed externally
        self._load_cache()
        new_count = 0
        for topic in bagOfWords:
            if topic not in self.embeddingsCache:
                new_count += 1

        if new_count == 0:
            print("All embeddings already cached")
            return

        # Load the transformer model and calculate the overall topic embedding
        print("Importing transformer library...")
        from sentence_transformers import SentenceTransformer
        print("Loading transformer model...")
        emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Calculating embeddings...", end="", flush=True)

        new_count = 0
        for i, topic in enumerate(bagOfWords):
            if topic not in self.embeddingsCache:
                self.embeddingsCache[topic] = emb_model.encode(topic)
                new_count += 1
            if i and i % 1000 == 0:
                print(".", end="", flush=True)
                self._save_cache()
        print(f" done (new embeddings: {new_count} over {len(bagOfWords)})")
        self._save_cache()