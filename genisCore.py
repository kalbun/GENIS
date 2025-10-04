# import necessary libraries
import re
import spacy
# Assuming preprocessing is a module available in the environment
from preprocessing import ReviewPreprocessor

def genisCore(
    sentence: str,
    preprocessor: ReviewPreprocessor = None,
    nlp: spacy.language.Language = None
) -> tuple:
    """
    Core of GENIS algorithm.
    Given a sentence, extract adjective-noun pairs using SpaCy.
    Returns a tuple (pairs, nouns) where:
    - pairs: list of (noun, adjective) tuples
    - nouns: list of unique nouns found in the sentence
    """
    #
    # Start of GENIS core
    #
    if preprocessor is None:
        preprocessor = ReviewPreprocessor()
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    # If not, process the review to extract adjective-noun pairs.
    # split sentences on hard punctuation (periods, exclamation marks, question marks)
    pairs = []
    nouns = []
    # This regex splits on periods, exclamation marks, and question marks,
    sentences = re.split(r'(?<=[.!?]) +', sentence)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 4:
            continue
        # Process the sentence with SpaCy.
        # This is the core idea of the method: we assume that the sentiment in a review
        # is mainly expressed by nouns combined with adjectives, like in "good music"
        # or "awful service"
        # The extraction uses Spacy.
        # - "amod" means adjectival modifier (e.g., "good" in "good music")
        # - "acomp" means adjectival complement (e.g., "good" in "the product is good")
        # - "nsubj" means nominal subject (e.g., "product" in "the product is good") 
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ == "NOUN":
                # Token "children" are the words that depend on it.
                for child in token.children:
                    if child.dep_ == "amod":
                        # adjective modifier (e.g., "good" in "good music")
                        pairs.append((token.text, child.text))
                        nouns.append(token.text)
            elif token.dep_ == "acomp":
                # adjectival complement (e.g., "good" in "the product is good").
                # Now search its subject (the noun).
                subjects = [child for child in token.head.children if child.dep_ == "nsubj"]
                if subjects:
                    # Found, we can add the pair
                    pairs.append((subjects[0].text, token.text))
                    nouns.append(subjects[0].text)
        # for token in doc: (end)
    # for sentence in sentences: (end)

    # Lemmatization is useful for cases where singular and plural forms are used
    # interchangeably, like "good music" and "good musics".
    pairs = [(preprocessor.LemmatizeText(noun), adj) for noun, adj in pairs]
    # Remove duplicates from pairs
    pairs = sorted(list(set(pairs)))
    # Recalculate the nouns based on the pairs
    nouns = sorted(list(set([noun for noun, _ in pairs])))
    #
    # End of the GENIS core
    #
    return pairs, nouns
