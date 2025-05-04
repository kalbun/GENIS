import os
import json
import threading
import textwrap
import re
import warnings
from mistralai import SDKError, Mistral
from key import MistraAIKey as api_key

warnings.filterwarnings("ignore")

class Sentiments:
    def __init__(self, cachePath: str, bypass_cache: bool = False):
        """
        Initialize the Sentiments class with the given cache path and bypass cache option.
        The class is responsible for managing sentiment analysis using a language model.

        :param cachePath: the path to the cache directory        
        :param bypass_cache: whether to bypass the cache loading
        """
        # Global sentiment cache is maintained per file
        self.sentimentCache: dict = {}
        self.sentimentCacheFile: str = os.path.join(cachePath, "sentiment_cache.json")
        self.sentimentLogFile: str = os.path.join(cachePath, "sentiment_log.txt")
        #
        self.sentimentCacheBypass: bool = bypass_cache
        # Using a semaphore for thread-safe access
        self.sentimentCacheSemaphore = threading.Semaphore(1)
        # Using a semaphore for thread-safe access to the LLM
        # Limiting the number of concurrent requests to 6, because
        # Mistral has a limit of 6 concurrent requests per API key
        self.llmSemaphore = threading.Semaphore(6)
        # Initialize the Mistral client
        self.genAI_Client = Mistral(api_key=api_key)

        if not os.path.exists(cachePath):
            os.makedirs(cachePath, exist_ok=True)
        self.sentimentCache_Load()

    def sentimentCache_Load(self) -> dict:
        """
        Load the sentiment cache from the file if it exists and bypass is not set.
        
        :return: the sentiment cache as a dictionary
        """
        if (not self.sentimentCacheBypass) and os.path.exists(self.sentimentCacheFile):
            try:
                with open(self.sentimentCacheFile, "rt", encoding="utf-8") as f:
                    self.sentimentCache = json.load(f)
            except Exception:
                pass
        return self.sentimentCache

    def sentimentCache_Save(self):
        """
        Save the sentiment cache to the file if bypass is not set.
        """
        if not self.sentimentCacheBypass:
            if self.sentimentCacheSemaphore.acquire():
                try:
                    with open(self.sentimentCacheFile, "wt", encoding="utf-8") as f:
                        json.dump(self.sentimentCache, f, ensure_ascii=False, indent=4)
                except Exception:
                    pass
                self.sentimentCacheSemaphore.release()

    def sentimentCache_CreateItem(self, item: str, topic_sentiments: dict):
        """
        Create a new item in the sentiment cache with the given topic sentiments.
        
        :param item: the item to create in the cache
        :param topic_sentiments: the topic sentiments to associate with the item
        """
        if self.sentimentCacheSemaphore.acquire():
            self.sentimentCache[item] = {}
            if topic_sentiments:
                # If topic sentiments are provided, add them to the cache
                self.sentimentCache[item]['sentiments'] = topic_sentiments
            self.sentimentCacheSemaphore.release()

    def sentimentCache_updateSentiment(self, item: str, topic_sentiments: dict):
        """
        Update the sentiment for the given item in the cache.
        
        :param item: the item to update in the cache
        :param topic_sentiments: the topic sentiments to update
        """
        if self.sentimentCacheSemaphore.acquire():
            # Update or create the item with the provided sentiments
            self.sentimentCache[item] = {'sentiments': topic_sentiments}
            self.sentimentCacheSemaphore.release()

    def sentimentCache_updateOriginalRating(self, item: str, originalRating: float, newRating: float):
        """
        Update the original rating in the sentiment cache for the given item.
        
        :param item: the item to update in the cache
        :param originalRating: the original rating to update
        :param newRating: the new rating to set in the cache
        """
        if self.sentimentCacheSemaphore.acquire():
            self.sentimentCache[item][str(originalRating)] = newRating
            self.sentimentCacheSemaphore.release()

    def adjustRating(self, original_score, topic_sentiments):
        # Calculate the adjusted rating based on numeric sentiment values
        numeric_values = [v for v in topic_sentiments.values() if isinstance(v, (int, float))]
        if numeric_values:
            return original_score + sum(numeric_values)
        return original_score

    def returnMostRelevantTopic(self, topics: list[str]) -> str:
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

        model: str = "mistral-small-latest"

        if self.llmSemaphore.acquire():
            try:
                response = self.genAI_Client.chat.complete(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )
            except SDKError as e:
                self.llmSemaphore.release()
                raise e
            with open(self.sentimentLogFile, "a") as f:
                f.write(f"Prompt: {topics}\n")
                f.write(f"Response: {response.choices[0].message.content.strip()}\n")
            self.llmSemaphore.release()
            # The response should be a single line with the topic.
            topic = response.choices[0].message.content.strip()

        return topic

    def topicsFromSentence(self, sentence: str) -> list[str]:
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

        model: str = "mistral-small-latest"

        if self.llmSemaphore.acquire():
            try:
                response = self.genAI_Client.chat.complete(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )
            except SDKError as e:
                self.llmSemaphore.release()
                raise e
            self.llmSemaphore.release()
            topics = response.choices[0].message.content.strip().split("\n")
        return topics

    def aggregateSimilarTopics(self, topics: list[str]) -> list[str]:
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
            ---
        """)

        model: str = "mistral-small-latest"

        if self.llmSemaphore.acquire():
            try:
                response = self.genAI_Client.chat.complete(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )
            except SDKError as e:
                self.llmSemaphore.release()
                raise e
            self.llmSemaphore.release()
            # attempt to parse the response. Items should be separated by newlines
            # and should be put in a list.
            aggregated_sentiments = response.choices[0].message.content.strip().split("\n")
            # Remove empty strings and strip whitespace from each item
            aggregated_sentiments = [item.strip() for item in aggregated_sentiments if item.strip()]
        return aggregated_sentiments

    def getTypicalTopics(self, generalTopic: str) -> list[str]:
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

        model: str = "mistral-small-latest"

        if self.llmSemaphore.acquire():
            try:
                response = self.genAI_Client.chat.complete(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )
            except SDKError as e:
                self.llmSemaphore.release()
                raise e
            self.llmSemaphore.release()
            # attempt to parse the response. Items should be separated by newlines
            # and should be put in a list.
            associatedSentiments = response.choices[0].message.content.strip().split("\n")
            associatedSentiments.append(generalTopic)
        return associatedSentiments

    def assignGradeToReview(self, review: str) -> int:
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
                Read this Amazon review and rate experience
                from 1 to 10, where 1 worst experience, 10 best.
                You can use half scores like 6.5.
                Be moderate with scores.
                RETURN ONLY SCORE. NO COMMENTS OR ANYTHING ELSE!!!
                ---
                Review to score:
                {review}""")

        while (retry_counter < 3):
            try:
                if self.llmSemaphore.acquire():
                    try:
                        response = self.genAI_Client.chat.complete(
                            model=model,
                            temperature=0.0,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ]
                        )
                    except SDKError as e:
                        self.llmSemaphore.release()
                        raise e
                    self.llmSemaphore.release()
                    score = float(response.choices[0].message.content.strip())
                    break
            except SDKError:
                retry_counter += 1
                continue
            except Exception:
                # exit
                print("E", end="", flush=True)
                break  # empty output

        return score

    def parseScore(self, text: str, topics: list[str]) -> tuple[dict, str]:
        """
        Parse the sentiment of the review text using the LLM model.
        
        :param text: the review text
        :param topics: the topics to consider for sentiment analysis
        :return: a tuple of (topic sentiment dictionary, return state)
        The return state can be:
            - "C": cached data
            - "_": data parsed from the LLM
            - "J": JSON parsing error
            - "E": exception occurred
        """
        output: dict = {}
        retry_counter: int = 0
        returnState: str = "N"
        prompt: str = ""
        
        #    if not any(topic.lower() in text.lower() for topic in topics):
        #        return output
        
        prompt = textwrap.dedent(f"""
                Your task is to
                1) search list of topics in a review text
                2) calculate sentiment of each topic in text as score
                    -1 (negative), 0 (neutral/not found), 1 (positive).
                3) return sentiments as JSON object. ONLY JSON. No comments, explanations, anything else.

                Example 1:
                    text: 'Not bad CD, not great either'
                    Topics: ['cd','cover']
                    Output:
                    {{
                        "cd": 0,
                        "cover": 0
                    }}
                    Why: cd neutral, cover not found.

                Example 2:
                    text: 'Gosh, the cd is... bombastic! But the cover is awful :-('
                    Topics: ['cd','cover']
                    Output:
                    {{
                        "cd": 1,
                        "cover": -1
                    }}
                    Why: cd positive, cover negative.

                ---
                TEXT:
                {text}.
                ---
                TOPICS:
                {topics}
                ---""")

        model: str = "mistral-small-latest"

        # if the text is cached, retrieve the sentiment from the cache
        # and return it
        if text in self.sentimentCache:
            cached_data = self.sentimentCache[text]
            if 'sentiments' in cached_data:
                # if sentiments are cached, check if adjusted rating is cached
                output = cached_data['sentiments']
                return output, 'C'

        while (retry_counter < 3):
            try:
                if self.llmSemaphore.acquire():
                    try:
                        response = self.genAI_Client.chat.complete(
                            model=model,
                            temperature=0.0,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ]
                        )
                    except SDKError as e:
                        self.llmSemaphore.release()
                        raise e
                    with open(self.sentimentLogFile, "a") as f:
                        f.write(f"Text: {text}\n")
                        f.write(f"Topics: {topics}\n")
                        f.write(f"Response: {response.choices[0].message.content.strip()}\n")
                    self.llmSemaphore.release()
                    # attempt to parse the response
                    try:
                        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
                        if match:
                            output = json.loads(match.group(0))
                            # update the sentiment cache with the parsed output
                            self.sentimentCache_updateSentiment(text, output)
                            self.sentimentCache_Save()
                            returnState = "_"
                    except json.JSONDecodeError:
                        returnState = "J"
                    break
            except SDKError:
                retry_counter += 1
                continue
            except Exception as e:
                # exit
                returnState = "E"
                break  # empty output

        return output, returnState

    def sentimentCache_getSentimentAndAdjustedRating(self, text: str, original_rating: float, topics: list[str], forceRandom: bool = False) -> tuple[dict, float]:
        """
        Retrieve sentiment data and calculate the adjusted rating based on available topics.
        
        :param text: the review text
        :param original_rating: the original rating of the review
        :param topics: topics for sentiment analysis
        :param forceRandom: if set, adjust rating randomly
        :return: a tuple of (topic sentiment dictionary, adjusted rating)
        """
        topic_sentiments: dict = {}
        adjusted_rating: float = original_rating
        import random
        if forceRandom:
            # Case 1: adjusted rating is random
            for i in range(len(topics)):
                adjusted_rating += round(random.random() * 4 - 2) / 2
        elif text in self.sentimentCache:
            # Case 2: sentence already cached
            cached_data = self.sentimentCache[text]
            # if text in cache, check if sentiments are cached
            if 'sentiments' in cached_data:
                # if sentiments are cached, check if adjusted rating is cached
                topic_sentiments = cached_data['sentiments']
                if str(original_rating) in cached_data and cached_data[str(original_rating)]:
                    # if adjusted rating is cached, there is nothing to do
                    # adjusted_rating = cached_data[str(original_rating)]
                    print("C", end="", flush=True)
                else:
                    # if adjusted rating is not cached, calculate it based on the sentiments
                    adjusted_rating = self.adjustRating(original_rating, topic_sentiments)
                    self.sentimentCache_updateOriginalRating(text, original_rating, adjusted_rating)
                    self.sentimentCache_Save()
                    print(".", end="", flush=True)
            else:
                # if sentiments are not cached and there are topics, calculate them
                if len(topics) > 0:
                    topic_sentiments = self.parseScore(text, topics)
                    self.sentimentCache_updateSentiment(text, topic_sentiments)
                    topic_sentiments, adjusted_rating = self.sentimentCache_getSentimentAndAdjustedRating(text, original_rating, topics)
                else:
                    # Sentiment empty and no topics for sentiment analysis.
                    self.sentimentCache_updateOriginalRating(text, original_rating, original_rating)
                    self.sentimentCache_Save()
                    print("_", end="", flush=True)
        else:
            # if text not in cache, create a new entry and recurse
            self.sentimentCache_CreateItem(text, {})
            topic_sentiments, adjusted_rating = self.sentimentCache_getSentimentAndAdjustedRating(text, original_rating, topics)

        return topic_sentiments, adjusted_rating