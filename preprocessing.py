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

_PREPROCESSING_CACHE: dict = {}
_PREPROCESSING_CACHE_FILE: str = ""

def preprocessingCache_init(cache_file: str):
    """
    Initialize the preprocessing cache and load it from the specified file.
    
    Args:

    cache_file: str - the path to the cache file.
    """
    global _PREPROCESSING_CACHE, _PREPROCESSING_CACHE_FILE
    _PREPROCESSING_CACHE_FILE = cache_file
    _PREPROCESSING_CACHE = {}
    preprocessingCache_load()

def preprocessingCache_load() -> dict:
    """
    Load the preprocessing cache from the file if it exists.
    
    Returns:
    dict: the preprocessing cache as a dictionary.
    """
    global _PREPROCESSING_CACHE, _PREPROCESSING_CACHE_FILE
    if os.path.exists(_PREPROCESSING_CACHE_FILE):
        try:
            with open(_PREPROCESSING_CACHE_FILE, "r", encoding="utf-8") as f:
                _PREPROCESSING_CACHE = json.load(f)
        except Exception:
            pass
    return _PREPROCESSING_CACHE

def preprocessingCache_save():
    """
    Save the preprocessing cache to the file.
    """
    global _PREPROCESSING_CACHE, _PREPROCESSING_CACHE_FILE
    try:
        with open(_PREPROCESSING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_PREPROCESSING_CACHE, f, ensure_ascii=False, indent=4)
    except Exception:
        pass

def preprocess_reviews(reviews: list[str]) -> list[str]:
    """
    Preprocess the reviews by tokenizing, removing stop words, and lemmatizing.

    :param reviews: list[str] - the list of reviews to preprocess.
    :return: list[str] - the list of preprocessed reviews.
    """

    def get_wordnet_pos(tag: str) -> str:
        """
        Convert POS tag to WordNet POS tag.
        
        Args:
        tag: str - the POS tag to convert.

        Returns:
        str: the corresponding WordNet POS tag.
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

    preprocessed_sentences: list[str] = []
    correction_cache: dict[str, str] = {}

    stop_words: set = set(stopwords.words('english'))
    lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
    spell_checker: SpellChecker = SpellChecker()
    spell_checker.word_frequency.load_words(stop_words)

    tokens: list[list[str]] = []
    pos_tags: list[tuple[str, str]] = []

    bagOfWords: dict[str, int] = {}

    _CORRECTION_CACHE_FILE = os.path.join(os.path.dirname(__file__), "correction_cache.json")
    if os.path.exists(_CORRECTION_CACHE_FILE):
        try:
            with open(_CORRECTION_CACHE_FILE, "r", encoding="utf-8") as f:
                correction_cache = json.load(f)
        except Exception:
            pass

    # Set lowercase, remove duplicates and sort the reviews
    reviews = [sentence.lower() for sentence in reviews]
    # remove punctuation and special characters
    reviews = sorted(set([re.sub(r'\W+', ' ', sentence) for sentence in reviews]))
    # create a list of unique words from the reviews
    bagOfWords = sorted(set({word for sentence in reviews for word in sentence.split()}))
    # Now do spell correction for the bag of words
    for word in bagOfWords:
        if word not in correction_cache:
            corrected_word: str = spell_checker.correction(word)
            if corrected_word != None and corrected_word != word:
                correction_cache[word] = corrected_word
    del bagOfWords
    # Save the correction cache to file
    try:
        with open(_CORRECTION_CACHE_FILE, "wt", encoding="utf-8") as f:
            json.dump(correction_cache, f, ensure_ascii=False, indent=4)
    except Exception:
        pass

    updated_reviews = []
    for review in reviews:
        updates_sentence: str = ""
        # Do grammar spell word by word, replacing on the fly
        for word in review.split():
            # also remove words with digits and punctuation
            if any(char.isdigit() for char in word):
                continue
            if word in correction_cache:
                word = correction_cache[word]
            updates_sentence += word + " "
        updated_reviews.append(updates_sentence.strip())
    reviews = updated_reviews

    for idx, review in enumerate(reviews):
        if idx % 1000 == 0:
            print(".", end="", flush=True)
        # Check if the sentence is already in the cache
        if review in _PREPROCESSING_CACHE:
            preprocessed_sentences.append(_PREPROCESSING_CACHE[review])
            continue
        # Tokenize the sentence, apply POS tagging and lemmatization
        tokens = word_tokenize(review)
        pos_tags = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                            for word, tag in pos_tags]

        # Keep only nouns and adjectives
        filtered_tokens = [
            word for word,tag in zip(lemmatized_tokens,pos_tags)
            if (tag[1].startswith('N')or tag[1].startswith('V'))
            and word not in stop_words
            and len(word) > 1
        ]
        filtered_tokens = list(set(filtered_tokens))

        # Rebuild the sentence
        filtered_sentence = " ".join(filtered_tokens)
        # Save to cache
        _PREPROCESSING_CACHE[review] = filtered_sentence
        preprocessingCache_save()
        # Append to the list of preprocessed sentences
        preprocessed_sentences.append(filtered_sentence)
    return preprocessed_sentences

def load_reviews(file_path: str, max_reviews: int, label_text: str, label_rating: str, seed: int) -> tuple[list[dict], set[int]]:
    """
    Load reviews from a file and return a list of dictionaries containing the text and overall rating.
    
    Args:
    file_path: str - the path to the file containing the reviews.
    max_reviews: int - the maximum number of reviews to load.
    label_text: str - label of field containing the review text.
    label_rating: str - label of field containing the review rating.
    seed: int - seed for random number generation.

    Returns:
    tuple[list[dict], set[int]]: a list of dictionaries containing the text and
        overall rating of the reviews, and the set of random indices to which
        the reviews correspond.
    """
    reviews = []
    total_lines = 0

    with open(os.path.join(os.path.dirname(__file__),file_path), "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    random.seed(seed)
    random_indices = set(random.sample(range(total_lines), min(max_reviews, total_lines)))
    selected_lines = []
    with open(os.path.join(os.path.dirname(__file__),file_path), "r", encoding="utf-8") as f:
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
    print(f"\nLoaded {len(reviews)} reviews.")
    return reviews, random_indices
