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

    if not any(topic.lower() in text.lower() for topic in topics):
        return output

    prompt = textwrap.dedent(f"""
            Your task is to
            1) search a list of topics in a review text
            2) calculate sentiment of each topic in the text as score -1, -0.5, 0, 0.5, 1
            3) return the sentiment for each topic in the list as a JSON object.

            Example text:
            'This is a great CD. I love it.'
            Example topics:
            ['cd','cover']
            Example output:
            {{
                "cd": 1,
                "cover": 0
            }}

            Return ONLY AND EXCLUSIVELY a JSON. I do not want comments,
            explanations, or anything else. I want ONLY the JSON object.

            TEXT:
            {text}
            ---
            TOPICS:
            {topics}
            ---""")

    model = "mistral-small-latest"

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
                # attempt to parse the response
                try:
                    match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
                    if match:
                        output = json.loads(match.group(0))
                        print(".", end="", flush=True)
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
        if 'sentiments' in cached_data:
            # Sentiment already cached
            if cached_data['sentiments']:
                # Sentiment cached and not empty
                topic_sentiments = cached_data['sentiments']
                if str(original_rating) in cached_data:
                    adjusted_rating = cached_data[str(original_rating)]
                else:
                    # Adjusted rating not found in cache, nothing to do
                    pass
            else:
                # Sentiment cached but empty. Nothing to do.
                pass
            print("C", end="", flush=True)
        else:
            # Text cached but sentiment not found. Calculate it and the corrected rating.
            topic_sentiments = sentiment_parseScore(text, topics)
            adjusted_rating = sentiment_adjustRating(original_rating, topic_sentiments)
            sentimentCache_updateOriginalRating(text, original_rating, adjusted_rating)
            sentimentCache_Save()
            print(".", end="", flush=True)
    else:
        # Case 3: sentiment not cached, create a new entry and recurse
        sentimentCache_CreateItem(text, {})
        sentimentCache_Save()
        topic_sentiments, adjusted_rating = sentimentCache_getSentimentAndAdjustedRating(text, original_rating, topics)
    return topic_sentiments, adjusted_rating
