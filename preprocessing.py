import os
import json
import random
import re
from spellchecker import SpellChecker
import nltk

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if missing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class ReviewPreprocessor:
    """
    Class for preprocessing and correcting review text.
    
    Provides methods to load and save caches for tokenization and correction,
    preprocess reviews by tokenizing, removing stop words, lemmatizing, and perform spell correction,
    and load reviews from a file.
    """

    def __init__(self, file_prefix: str):
        """
        Initialize the preprocessor with cache files based on a file prefix.

        Args:
            file_prefix (str): The prefix for the cache file names.
        """
        self.preprocessing_cache: dict = {}
        self.preprocessing_cache_file: str = file_prefix + "_preprocessing_cache.json"
        self.correction_cache: dict = {}
        self.correction_cache_file: str = file_prefix + "_correction_cache.json"
        self._load_preprocessing_cache()
        self._load_correction_cache()

    # ---------------------- Cache Methods ---------------------- #
    def _load_preprocessing_cache(self) -> dict:
        """
        Loads the preprocessing cache from disk if the file exists.

        Returns:
            dict: The loaded preprocessing cache.
        """
        if os.path.exists(self.preprocessing_cache_file):
            try:
                with open(self.preprocessing_cache_file, "r", encoding="utf-8") as f:
                    self.preprocessing_cache = json.load(f)
            except Exception:
                pass
        return self.preprocessing_cache

    def _save_preprocessing_cache(self):
        """
        Saves the current preprocessing cache to disk.
        """
        try:
            with open(self.preprocessing_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.preprocessing_cache, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    def _load_correction_cache(self) -> dict:
        """
        Loads the correction cache from disk if the file exists.

        Returns:
            dict: The loaded correction cache.
        """
        if os.path.exists(self.correction_cache_file):
            try:
                with open(self.correction_cache_file, "r", encoding="utf-8") as f:
                    self.correction_cache = json.load(f)
            except Exception:
                pass
        return self.correction_cache

    def _save_correction_cache(self):
        """
        Saves the current correction cache to disk.
        """
        try:
            with open(self.correction_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.correction_cache, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    # ---------------------- Preprocessing Methods ---------------------- #
    @staticmethod
    def _get_wordnet_pos(tag: str) -> str:
        """
        Convert POS tag to WordNet POS tag.

        Args:
            tag (str): The POS tag to convert.
        
        Returns:
            str: The corresponding WordNet POS tag.
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    def preprocess_reviews(self, reviews: dict[str, float]) -> tuple[dict[str, dict], dict[str, float]]:
        """
        Preprocess reviews by tokenizing, removing stop words,
        and lemmatizing. Also applies spell correction.

        The method returns two dictionaries:
            - preprocessed_reviews: with tokens and scores.
            - corrected_reviews: with the updated sentences after spell checking.

        Args:
            reviews (dict[str, float]): Reviews to preprocess with their scores.
        
        Returns:
            tuple: Preprocessed reviews and corrected reviews.
        """
        preprocessed_reviews: dict[str, dict] = {}
        corrected_reviews: dict[str, float] = {}
        word_correction_cache: dict[str, str] = {}

        # Initialize resources
        stop_words: set = set(stopwords.words('english'))
        lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        spell_checker: SpellChecker = SpellChecker()
        spell_checker.word_frequency.load_words(stop_words)

        for review in reviews.keys():

            if (review in self.correction_cache) and (self.correction_cache[review] == reviews[review]):
                # If review already processed, use cached tokens
                corrected_reviews[review] = reviews[review]
                continue

            # Remove punctuation and special characters
            _reviews = {key.lower(): value for key, value in reviews.items()}
            reviews = _reviews.copy()
            _reviews = {re.sub(r'\W+', ' ', key): value for key, value in reviews.items()}
            reviews = _reviews.copy()
            del _reviews

            # Create a sorted list of unique words from all review texts
            bag_of_words = sorted(set(word for value in reviews.keys() for word in value.split()))
        
        # Perform spell correction for each word in the bag of words
        for word in bag_of_words:
            if word not in word_correction_cache:
                corrected_word: str = spell_checker.correction(word)
                if corrected_word is not None and corrected_word != word:
                    word_correction_cache[word] = corrected_word
        del bag_of_words

        # Rebuild reviews with corrected words
        for review, score in reviews.items():
            updated_sentence: str = ""

            # If review already processed, use cached tokens
            if review in self.correction_cache:
                corrected_reviews[review] = score
                continue

            # If review already processed, use cached tokens
            # Replace word by word after correcting and removing words with digits
            for word in review.split():
                if any(char.isdigit() for char in word):
                    continue
                if word in word_correction_cache:
                    word = word_correction_cache[word]
                updated_sentence += word + " "
            corrected_reviews[updated_sentence.strip()] = score
            # Saves a corrected version of the review in the cache
            self.correction_cache[updated_sentence.strip()] = score
        reviews = corrected_reviews

        # Process each corrected review
        for idx, (review, score) in enumerate(reviews.items()):
            if idx and (idx % 500 == 0):
                print(".", end="", flush=True)
            # If review already processed, use cached tokens
            if review in self.preprocessing_cache:
                preprocessed_reviews[review] = self.preprocessing_cache[review]
                continue
            # Tokenize, POS tag, and lemmatize the review
            tokens = word_tokenize(review)
            pos_tags = pos_tag(tokens)
            lemmatized_tokens = [
                lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
                for word, tag in pos_tags
            ]
            # Filter tokens: keep only nouns and verbs, remove stop words and short words
            filtered_tokens = [
                word for word, tag in zip(lemmatized_tokens, pos_tags)
                if (tag[1].startswith('N') or tag[1].startswith('V'))
                and word not in stop_words
                and len(word) > 1
            ]
            # Remove duplicates by converting to a set and then back to list
            filtered_tokens = list(set(filtered_tokens))
            # Rebuild the sentence from filtered tokens
            filtered_sentence = " ".join(filtered_tokens)
            # Save result in cache and to return dictionary
            self.preprocessing_cache[review] = {"tokens": filtered_sentence, "score": score}
            self._save_preprocessing_cache()
            preprocessed_reviews[review] = {"tokens": filtered_sentence, "score": score}
        return preprocessed_reviews, corrected_reviews

    # ---------------------- Review Loader ---------------------- #
    def load_reviews(self, file_path: str, max_reviews: int, label_text: str, label_rating: str, seed: int) -> tuple[list[dict], set[int]]:
        """
        Load reviews from a file and return a list of dictionaries containing review text and overall rating.
        Also returns the set of random indices corresponding to selected reviews.

        Args:
            file_path (str): Path to the file with reviews.
            max_reviews (int): Maximum number of reviews to load.
            label_text (str): Field name for review text.
            label_rating (str): Field name for review rating.
            seed (int): Seed for random number generation.

        Returns:
            tuple: A list of review dictionaries and a set of selected review indices.
        """
        reviews = []
        total_lines = 0

        full_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(full_path, "r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
        random.seed(seed)
        random_indices = set(random.sample(range(total_lines), min(max_reviews, total_lines)))
        selected_lines = []
        with open(full_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in random_indices:
                    selected_lines.append(line)
                if len(selected_lines) >= max_reviews:
                    break

        for line in selected_lines:
            try:
                data = json.loads(line)
                if label_text in data and label_rating in data:
                    reviews.append({
                        "text": data[label_text].replace('<br />', '\n').replace('"', "'").strip().lower(),
                        "overall": data[label_rating]
                    })
            except json.JSONDecodeError:
                continue
        return reviews, random_indices