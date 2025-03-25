import os
import json
import threading
import re
import warnings
from mistralai import SDKError, Mistral
from key import MistraAIKey as api_key

warnings.filterwarnings("ignore")

# Global sentiment cache is maintained per file
sentimentCache: dict = {}
sentimentCacheFile: str = ""

# Initialize the Mistral client
genAI_Client = Mistral(api_key=api_key)

# Global sentiment cache is maintained per file
def sentimentCache_Load(cache_file: str) -> dict:

    global sentimentCache
    global sentimentCacheFile

    sentimentCacheFile = cache_file
    sentimentCache = {}
    if os.path.exists(sentimentCacheFile):
        try:
            with open(sentimentCacheFile, "rt", encoding="utf-8") as f:
                sentimentCache = json.load(f)
        except Exception:
            pass
    return sentimentCache

# Using a semaphore for thread-safe access
cacheSemaphore = threading.Semaphore(1)

def sentimentCache_Save(cache: dict):
    if (cacheSemaphore.acquire()):
        try:
            with open(sentimentCacheFile, "wt", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=4)
        except Exception:
            pass
        cacheSemaphore.release()

llmSemaphore = threading.Semaphore(6)

def sentimentCache_CreateItem(cache: dict, item: str, topic_sentiments: dict):
    if cacheSemaphore.acquire():
        cache[item] = {'sentiments': topic_sentiments}
        cacheSemaphore.release()

def sentimentCache_updateOriginalRating(cache: dict, item: str, originalRating: float, newRating: float):
    if cacheSemaphore.acquire():
        cache[item][str(originalRating)] = newRating
        cacheSemaphore.release()

def adjust_rating(original_score, topic_sentiments):
    numeric_values = [v for v in topic_sentiments.values() if isinstance(v, (int, float))]
    if numeric_values:
        return original_score + sum(numeric_values)
    return original_score

def parseReviewSentimentLLM(text: str, topics: list[str]) -> dict:
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

    prompt = f"""Read TEXT:
            {text}
            ---
            For each topic in this LIST:
                {topics}
            1) check if topic is in TEXT
            2) if yes, return topic sentiment as -1, -0.5, 0, 0.5, 1 
            Return ***ONLY*** JSON like:
            {{
                "topic1": 0.5,
                "topic2": -0.5
            }}"""

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

def sentimentCache_getSentiment_and_AdjustedRating(text: str, original_rating: float, topics: list[str]) -> tuple[dict, float]:

    global sentimentCache

    topic_sentiments: dict = {}
    adjusted_rating: float = 0
    
    if text in sentimentCache:
        cached_data = sentimentCache[text]
        if 'sentiments' in cached_data and cached_data['sentiments']:
            topic_sentiments = cached_data['sentiments']
        else:
            topic_sentiments = parseReviewSentimentLLM(text, topics)
            sentimentCache_CreateItem(sentimentCache, text, topic_sentiments)
            sentimentCache_Save(sentimentCache)
            # You would save cache here if needed
        if str(original_rating) in cached_data:
            print("C", end="", flush=True)
            return topic_sentiments, cached_data[str(original_rating)]
        else:
            adjusted_rating = adjust_rating(original_rating, topic_sentiments)
            sentimentCache_updateOriginalRating(sentimentCache, text, original_rating, adjusted_rating)
            sentimentCache_Save(sentimentCache)
    else:
        sentimentCache_CreateItem(sentimentCache, text, {})
        topic_sentiments, adjusted_rating = sentimentCache_getSentiment_and_AdjustedRating(text, original_rating, topics)
    return topic_sentiments, adjusted_rating
