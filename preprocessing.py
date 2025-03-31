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

def preprocess_reviews(reviews: list[str]) -> list[str]:
    """
    Preprocess the reviews by tokenizing, removing stop words, and lemmatizing.
 
    Args:
    reviews: list[str] - the list of reviews to preprocess.

    Returns:
    list[str]: the preprocessed reviews.
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

    for idx, review in enumerate(reviews):
        review = review.lower() if isinstance(review, str) else " "
        if idx % 1000 == 0:
            print(".", end="", flush=True)
        sentences = [review]
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            # Remove punctuation, non-alphanumerics, words with digits, and stop words in one pass.
            cleaned_tokens = []
            for token in tokens:
                if token.isalnum():
                    token = re.sub(r'\W+', '', token)
                    if not any(char.isdigit() for char in token) and token not in stop_words:
                        cleaned_tokens.append(token)
            # Remove duplicates
            unique_tokens = list(set(cleaned_tokens))
            
            # Correct spelling on unique tokens
            corrected_tokens: list[str] = []
            for token in unique_tokens:
                if 0 and (token not in correction_cache):
                    correctedToken: str = spell_checker.correction(token)
                    if correctedToken != None:
                        correction_cache[token] = correctedToken
                        token = correctedToken
                corrected_tokens.append(token)

            # Apply POS tagging and lemmatization
            pos_tags = pos_tag(corrected_tokens)
            lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                                for word, tag in pos_tags]
            lemmatized_tokens = list(set(lemmatized_tokens))  # Remove duplicates

            # Keep only nouns and adjectives
            filtered_tokens = [word for word, pos in pos_tag(lemmatized_tokens)
                            if pos.startswith('NN') or pos.startswith('JJ')]

            # Remove short words and stop words
            final_tokens = [word for word in filtered_tokens
                            if len(word) > 1 and word not in stop_words]
            final_tokens = list(set(final_tokens))  # Final deduplication

            # Rebuild the sentence
            filtered_sentence = " ".join(final_tokens)
            preprocessed_sentences.append(filtered_sentence)
    return preprocessed_sentences

def load_reviews(file_path: str, max_reviews: int, label_text: str, label_rating: str) -> tuple[list[dict], set[int]]:
    """
    Load reviews from a file and return a list of dictionaries containing the text and overall rating.
    
    Args:
    file_path: str - the path to the file containing the reviews.
    max_reviews: int - the maximum number of reviews to load.
    label_text: str - label of field containing the review text.
    label_rating: str - label of field containing the review rating.

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

def preprocess_and_extract_topics(reviews: list[dict]) -> list[str]:
    print("Preprocessing reviews...", end="")
    texts = [review["text"] for review in reviews]
    preprocessed_reviews = preprocess_reviews(texts)
    print(" completed.")
    # Return only preprocessed reviews – embedding extraction will occur in the embeddings module.
    return preprocessed_reviews