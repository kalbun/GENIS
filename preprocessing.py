print("\nStarting...")
import os
import json
import random
import html
import re
from spellchecker import SpellChecker
import nltk
import ftfy
import emoji

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix

# Download NLTK resources if missing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class ReviewPreprocessor:
    """
    A class to preprocess review texts by tokenizing, lemmatizing,
    correcting contractions, handling spelling, and filtering tokens.

    It supports saving and loading both preprocessing and correction caches.
    """

    def __init__(self, cachePath: str):
        """
        Initialize the preprocessor with file paths for caching results.

        Args:
            file_prefix (str): The prefix for the cache file names.
        """
        self.preprocessing_cache: dict = {}
        self.correction_cache: dict = {}
        self.preprocessing_cache_file: str = os.path.join(cachePath, "preprocessing_cache.json")
        self.correction_cache_file: str = os.path.join(cachePath, "correction_cache.json")
        self.LoadPreprocessingCache()
        self.LoadCorrectionCache()

    def __del__(self):
            # Save cache when the instance is being destroyed
            self.cacheSave()

    # ---------------------- Cache Methods ---------------------- #
    def LoadPreprocessingCache(self) -> dict:
        """
        Load the preprocessing cache from disk if it exists.

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

    def SavePreprocessingCache(self):
        """
        Save the current preprocessing cache to disk.
        """
        try:
            with open(self.preprocessing_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.preprocessing_cache, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    def LoadCorrectionCache(self) -> dict:
        """
        Load the correction cache from disk if it exists.

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

    def SaveCorrectionCache(self):
        """
        Save the current correction cache to disk, sorting the keys to ensure consistency.
        """
        try:
            self.correction_cache = dict(sorted(self.correction_cache.items()))
            with open(self.correction_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.correction_cache, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

    # ---------------------- Cache Get/Update Methods ---------------------- #
    def GetReviewFromCache(self, review: str) -> dict:
        """
        Retrieve a review from the preprocessing cache.

        Args:
            review (str): The review key in the cache.

        Returns:
            dict: The review data if found, otherwise None.
        """
        return self.preprocessing_cache.get(review)

    def AddSubitemToReviewCache(self, review: str, key: str, value) -> None:
        """
        Add or update a sub-item in the preprocessing cache for a given review.

        This method allows you to store additional computed data (such as bigrams or nouns)
        into the cache.

        Args:
            review (str): The review key in the cache.
            key (str): The sub-item key to add/update.
            value: The value to associate with the sub-item.
        """
        if review in self.preprocessing_cache:
            self.preprocessing_cache[review][key] = value
        else:
            self.preprocessing_cache[review] = {key: value}
        self.SavePreprocessingCache()

    def AddSubitemsToReviewCache(self, review: str, subitems: dict) -> None:
        """
        Add or update multiple sub-items in the preprocessing cache for a given review.

        Args:
            review (str): The review key in the cache.
            subitems (dict): A dictionary of sub-items to add/update.
        """
        if review in self.preprocessing_cache:
            self.preprocessing_cache[review].update(subitems)
        else:
            self.preprocessing_cache[review] = subitems
        self.SavePreprocessingCache()

    # ---------------------- Helper Methods ---------------------- #
    @staticmethod
    def GetWordnetPos(tag: str) -> str:
        """
        Convert POS tag to format recognized by WordNet.

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

    # ---------------------- Lemmatization Methods ---------------------- #
    def LemmatizeText(self, text: str) -> str:
        """
        Lemmatize a sentence or phrase by converting words to their base forms.
        For example, 'apples' becomes 'apple' and 'bad dogs' becomes 'bad dog'.

        Args:
            text (str): The text to lemmatize.

        Returns:
            str: The lemmatized text.
        """
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        lemmas = [lemmatizer.lemmatize(word, self.GetWordnetPos(tag))
                  for word, tag in pos_tags]
        return " ".join(lemmas)

    def LemmatizeList(self, texts: list[str]) -> list[str]:
        """
        Lemmatize a list of sentences or terms.

        Args:
            texts (list[str]): List of texts to lemmatize.

        Returns:
            list[str]: List of lemmatized texts.
        """
        return [self.LemmatizeText(text) for text in texts]

    # ---------------------- Preprocessing Methods ---------------------- #
    def PreprocessReviews(self, reviews: dict[str, float], callback = None) -> dict[str, dict]:
        """
        Preprocess review texts by correcting, tokenizing, lemmatizing, and spelling correction.

        The method applies the following steps:
        1. Correct contractions and perform lowercasing.
        2. unescape HTML and attempt to fix encoding issues.
        3. Generate a correction cache for misspelled words.
        4. Tokenize texts, apply POS tagging and lemmatization.
        5. Filter tokens by stopwords, token length, and word type.
        6. Update and save caches accordingly.

        Args:
            reviews (dict[str, float]): Dictionary where each key is a review text
                                        and value is its associated score.
            callback (function, optional): A callback function to be called after
            each review is processed. The function must accept two arguments:
                - index (int): The index of the review in the list.
                - total (int): The total number of reviews.

        Returns:
            dict[str, dict]: Mapping of original review texts to their processed data.
        """
        preprocessed_reviews: dict[str, dict] = {}

        # Initialize NLTK resources
        stop_words: set = set(stopwords.words('english'))
        spell_checker: SpellChecker = SpellChecker()
        spell_checker.word_frequency.load_words(stop_words)

        for rawReview in reviews:
            if rawReview in self.preprocessing_cache:
                continue

            # Apply a light preprocessing to make it more human-readable.
            readableReview = ftfy.fix_text(rawReview) # Take care of wrongly encoded characters
            readableReview = html.unescape(readableReview) # unescape HTML entities like &#34;
            readableReview = html.unescape(readableReview) # secord unescape to fix cases like &amp;#34;
            readableReview = (
                readableReview \
                .replace("\n", ". ")
                .replace("\r", ". ")
                .replace("\t", " ")
                .replace("<br />", " ")
            )
            # remove emojis - this proved to be more difficult than expected!
            readableReview = emoji.replace_emoji(readableReview, replace="")

            # Now do a heavier preprocessing of the reviews.
            # Note how we remove periods, to avoid splitting issues later on.
            correctedReview = fix(readableReview.lower())
            correctedReview = re.sub(r'[^A-Za-z0-9.?!]+', ' ', correctedReview)
            correctedReview = re.sub(r'([!?])\1+', r'\1', correctedReview)
            correctedReview = correctedReview.replace("mr.", "mister") \
                                             .replace("ms.", "miss") \
                                             .replace("dr.", "doctor") \
                                             .replace("etc.", "et cetera") \
                                             .replace("e.g.", "for example") \
                                             .replace("i.e.", "that is")

            if rawReview in self.preprocessing_cache:
                self.preprocessing_cache[rawReview].update({
                    'readable': readableReview,
                    'corrected': correctedReview,
                    'score': reviews[rawReview]
                })
            else:
                self.preprocessing_cache[rawReview] = {
                    'readable': readableReview,
                    'corrected': correctedReview,
                    'score': reviews[rawReview]
                }

        # Build a bag of unique words from all corrected sentences, ignoring numeric tokens.
        bag_of_words = sorted(
            set(
                word for review_data in self.preprocessing_cache.values()
                for word in re.findall(r'\b\w+\b', review_data['corrected'])
                if not any(char.isdigit() for char in word)
            )
        )

        # Update correction cache with spell corrections.
        for idx, word in enumerate(bag_of_words):
            if word not in self.correction_cache:
                corrected_word: str = spell_checker.correction(word)
                self.correction_cache[word] = corrected_word if corrected_word and corrected_word != word else word
                if idx and (idx % 100 == 0):
                    self.SaveCorrectionCache()
            # This is the longest operation
            if callback:
                callback(idx, len(bag_of_words))
        self.SaveCorrectionCache()

        # Update the corrected sentence based on the correction cache.
        for rawReview in reviews:
            correctedReview = self.preprocessing_cache[rawReview]["corrected"]
            self.preprocessing_cache[rawReview]["corrected"] = " ".join(
                self.correction_cache.get(word, word)
                for word in correctedReview.split()
#                if not any(char.isdigit() for char in word)
            ).strip()

        self.SavePreprocessingCache()

        # Return only preprocessed reviews that exist in the original input.
        preprocessed_reviews = {
            review: self.preprocessing_cache[review]
            for review in self.preprocessing_cache.keys()
            if review in reviews.keys()
        }

        return preprocessed_reviews

    # ---------------------- Review Loader ---------------------- #
    def LoadReviews(self, file_path: str, max_reviews: int,
                    label_text: str, label_rating: str, seed: int,
                    callback: None) -> tuple[list[dict], set[int]]:
        """
        Load reviews from a file and select a random subset.

        Reads the file line by line to count total lines, selects random indices,
        and then parses and filters the lines to load reviews based on the given labels.

        Args:
            file_path (str): Relative path to the file containing review JSON lines.
            max_reviews (int): Maximum number of reviews to return.
            label_text (str): Key name for the review text in the JSON.
            label_rating (str): Key name for the review rating in the JSON.
            seed (int): Random seed value for reproducibility.
            callback (function, optional): A callback function to be called after
            each review is processed. The function must accept two arguments:
                - index (int): The index of the review in the list.
                - total (int): The total number of reviews.

        Returns:
            tuple: A tuple containing:
                - A list of dictionaries with review text and rating.
                - A set of randomly selected indices corresponding to the reviews.
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
                        label_text: data[label_text].replace('<br />', '\n')
                                                 .replace('"', "'")
                                                 .strip().lower(),
                        label_rating: data[label_rating]
                    })
            except json.JSONDecodeError:
                continue
            if callback:
                callback(len(reviews), max_reviews)

        return reviews, random_indices