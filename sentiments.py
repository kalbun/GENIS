import os
import json
import threading
import textwrap
import re
import warnings
from mistralai import SDKError, Mistral
from key import MistraAIKey as api_key

warnings.filterwarnings("ignore")

# Global sentiment cache is maintained per file
sentimentCache: dict
sentimentCacheFile: str = ""
sentimentCacheBypass: bool = False

# Using a semaphore for thread-safe access
sentimentCacheSemaphore = threading.Semaphore(1)

sentimentLogFile: str = "sentiment.log"

# Using a semaphore for thread-safe access to the LLM
# Limiting the number of concurrent requests to 6, because
# Mistral has a limit of 6 concurrent requests per API key
llmSemaphore = threading.Semaphore(6)

# Initialize the Mistral client
genAI_Client = Mistral(api_key=api_key)

def sentimentCache_init(cache_file: str, bypass_cache: bool = False):
    """
    Initialize and load the sentiment cache variables. If bypass_cache is set
    to True, the cache remains empty and is not saved, but can still be used
    for storing sentiments.

    :param cache_file: the path to the cache file
    :param bypass_cache: whether to bypass the cache loading
    """
    global sentimentCache, sentimentCacheFile, sentimentCacheBypass

    sentimentCacheFile = cache_file
    sentimentCacheBypass = bypass_cache
    sentimentCache = {}
    sentimentCache_Load()

# Global sentiment cache is maintained per file
def sentimentCache_Load() -> dict:
    """
    Load the sentiment cache from the file if it exists and bypass is not set.
    
    :return: the sentiment cache as a dictionary
    """
    global sentimentCache, sentimentCacheFile, sentimentCacheBypass

    if (not sentimentCacheBypass) and os.path.exists(sentimentCacheFile):
        try:
            with open(sentimentCacheFile, "rt", encoding="utf-8") as f:
                sentimentCache = json.load(f)
        except Exception:
            pass
    return sentimentCache


def sentimentCache_Save():
    """
    Save the sentiment cache to the file if bypass is not set.
    """
    global sentimentCache, sentimentCacheFile, sentimentCacheBypass

    if (not sentimentCacheBypass):
        if (sentimentCacheSemaphore.acquire()):
            try:
                with open(sentimentCacheFile, "wt", encoding="utf-8") as f:
                    json.dump(sentimentCache, f, ensure_ascii=False, indent=4)
            except Exception:
                pass
            sentimentCacheSemaphore.release()


def sentimentCache_CreateItem(item: str, topic_sentiments: dict):
    """
    Create a new item in the sentiment cache with the given topic sentiments.
    
    :param item: the item to create in the cache
    :param topic_sentiments: the topic sentiments to associate with the item
    """
    global sentimentCache, sentimentCacheSemaphore

    if sentimentCacheSemaphore.acquire():
        sentimentCache[item] = {}
        if topic_sentiments:
            # If topic sentiments are provided, add them to the cache
            sentimentCache[item]['sentiments'] = topic_sentiments
        sentimentCacheSemaphore.release()

def sentimentCache_updateSentiment(item: str, topic_sentiments: dict):
    """
    Update the sentiment for the given item in the cache.

    :param item: the item to update in the cache
    :param topic_sentiments: the topic sentiments to update
    """
    global sentimentCache, sentimentCacheSemaphore

    if sentimentCacheSemaphore.acquire():
        # Update or create the item with the provided sentiments
        sentimentCache[item] = {'sentiments': topic_sentiments}
        sentimentCacheSemaphore.release()

def sentimentCache_updateOriginalRating(item: str, originalRating: float, newRating: float):
    """
    Update the original rating in the sentiment cache for the given item.

    :param item: the item to update in the cache
    :param originalRating: the original rating to update
    :param newRating: the new rating to set in the cache
    """
    global sentimentCache, sentimentCacheSemaphore

    if sentimentCacheSemaphore.acquire():
        sentimentCache[item][str(originalRating)] = newRating
        sentimentCacheSemaphore.release()

def sentiment_adjustRating(original_score, topic_sentiments):
    numeric_values = [v for v in topic_sentiments.values() if isinstance(v, (int, float))]
    if numeric_values:
        return original_score + sum(numeric_values)
    return original_score

def sentiment_returnMostRelevantTopic(topics: list[str]) -> str:
    """
    Return a term that describes all the topics in the list.
    
    :param topics: the list of topics to check
    :return: the most relevant topic from the list
    """

    prompt: str = ""
    topic: str = ""

    prompt = textwrap.dedent(f"""
        Return most representative topic from following list,
        Return ONLY topic. No comments, explanations, anything else!
        List:

        {topics}
    """)

    model = "mistral-small-latest"

    if (llmSemaphore.acquire()):
        try:
            response = genAI_Client.chat.complete(
                model = model,
                temperature=0.0,
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
        except SDKError as e:
            llmSemaphore.release()
            raise e
        with open(sentimentLogFile, "a") as f:
            f.write(f"Prompt: {topics}\n")
            f.write(f"Response: {response.choices[0].message.content.strip()}\n")
        llmSemaphore.release()
        # The response should be a single line with the topic.
        topic = response.choices[0].message.content.strip()

    return topic

def sentiment_topicsFromSentence(sentence: str) -> list[str]:
    """
    Return a list of topics from the given sentence.
    The topics extraction is based on the LLM model.

    :param sentence: the sentence to check
    :return: the list of topics from the sentence
    """
    topics: list[str] = []
    prompt: str = ""

    prompt = textwrap.dedent(f"""
        Read *Sentence* for emotional content.
        - return a list of newline-separated nouns associated with strong sentiments, empty string if no sentiment.
        - Return single word. 'cover' is OK, 'cd cover' is not.
        - Return ONLY nouns. No adjectives, verbs, adverbs, etc.
        - No comments, explanations, or anything else!

        Example with one strong sentiment (decent is a mild sentiment):
            *Sentence*:
                The cd is decent but the shipping was horrible!
            Answer:
                cover

        Example with two strong sentiments:
            *Sentence*:
                The cd is... bombastic!! but the shipping was horrible!
            Answer:
                cd
                shipping

        Example with no sentiment (good is a mild sentiment):
            *Sentence*:
                The cd is decent
            Answer (nothing, empty string):
                

        Here is *Sentence*:
        {sentence}
        """)

    model = "mistral-small-latest"

    if (llmSemaphore.acquire()):
        try:
            response = genAI_Client.chat.complete(
                model = model,
                temperature=0.0,
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
        except SDKError as e:
            llmSemaphore.release()
            raise e
        llmSemaphore.release()
        topics = response.choices[0].message.content.strip().split("\n")
    return topics


def sentiment_aggregateSimilarTopics(topics: list[str]) -> list[str]:
    """
    Aggregate similar topics in the sentiment dictionary.
    
    :param topics: the list of topics to aggregate
    :return: the aggregated topic sentiments dictionary
    """
    aggregated_sentiments: list[str] = []
    prompt: str = ""

    prompt = textwrap.dedent(f"""
        Your task is to read a list of term that can contain
        synonyms, and return a single word that represents the list.
        Return ONLY a term. No comments, explanations, or anything else!

        Example:

            List:
                disk
                cd
                album
                record
                
            Your output:
                disk

        List:
        {topics}
        ---""")

    model = "mistral-small-latest"

    if (llmSemaphore.acquire()):
        try:
            response = genAI_Client.chat.complete(
                model = model,
                temperature=0.0,
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
        except SDKError as e:
            llmSemaphore.release()
            raise e
        llmSemaphore.release()
        # attempt to parse the response. Items should be separated by newlines
        # and should be put in a list.
        aggregated_sentiments = response.choices[0].message.content.strip().split("\n")
        # Remove empty strings and strip whitespace from each item
        aggregated_sentiments = [item.strip() for item in aggregated_sentiments if item.strip()]    

    return aggregated_sentiments

def sentiment_getTypicalTopics(generalTopic: str) -> list[str]:
    """
    Return a list of topics typically associated with the given general topic.

    :param generalTopic: the general topic to check
    :return: the list of typical topics associated with the general topic
    """
    associatedSentiments: list[str] = []
    prompt: str = ""

    prompt = textwrap.dedent(f"""
        Task: return 5 terms typically associated with given general topic.
        Return ONLY term. No comments, explanations, or anything else!

        Example:
            general topic: 'digital music'
            Your output:
                ['music', 'sound', 'orchestra', 'band', 'concert']

        general topic:
        {generalTopic}
        """)

    model = "mistral-small-latest"

    if (llmSemaphore.acquire()):
        try:
            response = genAI_Client.chat.complete(
                model = model,
                temperature=0.0,
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
        except SDKError as e:
            llmSemaphore.release()
            raise e
        llmSemaphore.release()
        # attempt to parse the response. Items should be separated by newlines
        # and should be put in a list.
        associatedSentiments = response.choices[0].message.content.strip().split("\n")
        associatedSentiments.append(generalTopic)

    return associatedSentiments

def assignGradeToReview(review: str) -> int:
    """
    Assign a grade from 1 to 10 to the review, using a zero-shot
    LLM questioning. The grade is based only on the review text,
    because the LLM does not know the rating.

    :param review: the review text
    :param rating: the rating of the review
    """
    score: int = 0
    retry_counter: int = 0
    model: str = "mistral-small-latest"
    prompt: str = textwrap.dedent(f"""
            Read this Amazon review and rate the customer experience
            from 1 to 10, where 1 is the worst experience and 10 is the best.
            You can use half scores (e.g. 7.5).
            RETURN ONLY SCORE. NO COMMENTS, EXPLANATIONS, OR ANYTHING ELSE!!!
            ---
            Review to score:
            {review}""")

    while (retry_counter < 3):
        try:
            if (llmSemaphore.acquire()):
                try:
                    response = genAI_Client.chat.complete(
                        model = model,
                        temperature=0.0,
                        messages = [
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ]
                    )
                except SDKError as e:
                    llmSemaphore.release()
                    raise e
                llmSemaphore.release()
                score = int(response.choices[0].message.content.strip())
                break
        except SDKError:
            retry_counter += 1
            continue
        except Exception:
            # exit
            print("E", end="", flush=True)
            break # empty output

    return score



def sentiment_parseScore(text: str, topics: list[str]) -> dict:
    """
    Parse the sentiment of the review text using the LLM model.
    :param text: the review text
    :param topics: the topics to consider for sentiment analysis
    :return: the sentiments of the review text as a dictionary
    """

    output: dict = {}
    retry_counter: int = 0
    prompt: str = ""

#    if not any(topic.lower() in text.lower() for topic in topics):
#        return output

    prompt = textwrap.dedent(f"""
            Your task is to
            1) search list of topics in a review text
            2) calculate sentiment of each topic in text as score
                -1 (very negative), 0 (neutral/not found), 1 (very positive).
            3) return sentiments as JSON object. ONLY JSON. No comments, explanations, anything else.

            Example 1:
                text: 'Not bad CD, not great either'
                Topics: ['cd','cover']
                Output:
                {{
                    "cd": 0,
                    "cover": 0
                }}

            Example 2:
                text: 'Gosh, the cd is... bombastic! But the cover is awful :-('
                Topics: ['cd','cover']
                Output:
                {{
                    "cd": 1,
                    "cover": -1
                }}

            ---
            TEXT:
            {text}.
            ---
            TOPICS:
            {topics}
            ---""")

    model = "mistral-small-latest"

    # if the text is cached, retrieve the sentiment from the cache
    # and return it
    if text in sentimentCache:
        cached_data = sentimentCache[text]
        if 'sentiments' in cached_data:
            # if sentiments are cached, check if adjusted rating is cached
            output = cached_data['sentiments']
            print("C", end="", flush=True)
            return output

    while (retry_counter < 3):
        try:
            if (llmSemaphore.acquire()):
                try:
                    response = genAI_Client.chat.complete(
                        model = model,
                        temperature=0.0,
                        messages = [
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ]
                    )
                except SDKError as e:
                    llmSemaphore.release()
                    raise e
                with open(sentimentLogFile, "a") as f:
                    f.write(f"Text: {text}\n")
                    f.write(f"Topics: {topics}\n")
                    f.write(f"Response: {response.choices[0].message.content.strip()}\n")
                llmSemaphore.release()
                # attempt to parse the response
                try:
                    match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
                    if match:
                        output = json.loads(match.group(0))
                        # update the sentiment cache with the parsed output
                        sentimentCache_updateSentiment(text, output)
                        sentimentCache_Save()
                except json.JSONDecodeError:
                    print("J", end="", flush=True)
                break
        except SDKError:
            retry_counter += 1
            continue
        except Exception:
            # exit
            print("E", end="", flush=True)
            break # empty output

    return output

def sentimentCache_getSentimentAndAdjustedRating(text: str, original_rating: float, topics: list[str], forceRandom: bool = False) -> tuple[dict, float]:

    global sentimentCache
    global sentimentCacheSemaphore
    global sentimentCacheFile

    topic_sentiments: dict = {}
    adjusted_rating: float = original_rating
    import random
    if (forceRandom):
        # Case 1: adjusted rating is random
        for i in range(len(topics)):
            adjusted_rating += round(random.random() * 4 - 2)/2
    elif text in sentimentCache:
        # Case 2: sentemce already cached
        cached_data = sentimentCache[text]
        # if text in cache, check if sentiments are cached
        if 'sentiments' in cached_data:
            # if sentiments are cached, check if adjusted rating is cached
            topic_sentiments = cached_data['sentiments']
            if str(original_rating) in cached_data and cached_data[str(original_rating)]:
                # if adjusted rating is cached, there is nothing to do
#                adjusted_rating = cached_data[str(original_rating)]
                print("C", end="", flush=True)
            else:
                # if adjusted rating is not cached, calculate it based on the sentiments
                adjusted_rating = sentiment_adjustRating(original_rating, topic_sentiments)
                sentimentCache_updateOriginalRating(text, original_rating, adjusted_rating)
                sentimentCache_Save()
                print(".", end="", flush=True)
        else:
            # if sentiments are not cached and there are topics, calculate them
            if (len(topics) > 0):
                topic_sentiments = sentiment_parseScore(text, original_rating, topics)
                sentimentCache_updateSentiment(text, topic_sentiments)
                topic_sentiments, adjusted_rating = sentimentCache_getSentimentAndAdjustedRating(text, original_rating, topics)
            else:
                # Sentiment empty and not topics for sentiment analysis.
                sentimentCache_updateOriginalRating(text, original_rating, original_rating)
                sentimentCache_Save()
                print("_", end="", flush=True)

    else:
        # if text not in cache, create a new entry and recurse
        sentimentCache_CreateItem(text, {})
        topic_sentiments, adjusted_rating = sentimentCache_getSentimentAndAdjustedRating(text, original_rating, topics)

    return topic_sentiments, adjusted_rating
