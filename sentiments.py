import os
import json
import threading
import textwrap
import re
import warnings
import time  # Ensure time is imported
from mistralai import SDKError, Mistral
from key import MistraAIKey as api_key

warnings.filterwarnings("ignore")

class Sentiments:
    def __init__(self):
        """
        Initialize the Sentiments class.
        The class is responsible for managing sentiment analysis using a language model.
        """
        # Removed cache-related variables
        # Using a semaphore for thread-safe access to the LLM
        # Limiting the number of concurrent requests to 6, because
        # Mistral has a limit of 6 concurrent requests per API key
        self.llmSemaphore = threading.Semaphore(6)
        # Initialize the Mistral client
        self.genAI_Client = Mistral(api_key=api_key)

    # Removed __del__, cacheLoad, cacheSave, createItem, updateSentiment,
    # updateLLMscore, and updateOriginalRating as they are no longer needed.

    def invokeLLM(self, prompt: str, attempts: int = 5) -> tuple[str, bool]:
        """
        Invoke the LLM with the given prompt and data.
        """
        response: str = ""
        success: bool = False
        model: str = "mistral-small-latest"
        retry_counter: int = 0
        while retry_counter < attempts:
            try:
                with self.llmSemaphore:
                    response = self.genAI_Client.chat.complete(
                        model=model,
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}],
                    ).choices[0].message.content.strip()
                success = True
                break
            except SDKError:
                retry_counter += 1
                time.sleep(0.25)
            except Exception:
                break
        return response, success

    def gradeReview(self, review: str) -> tuple[float, str]:
        """
        Assign a grade from 1 to 10 to the review using zero-shot LLM questioning.
        """
        score: float = 0
        tag: str = "_"
        prompt: str = textwrap.dedent(f"""
                Read this ecommerce review and rate it from 1 to 10.
                Put yourself in customer clothes.
                1 = worst review, 10 = best.
                Can use half scores like 6.5 but not required to.
                RETURN ONLY SCORE. NO COMMENTS OR ANYTHING ELSE!!!
                ---
                Review:
                {review}""")
        response, success = self.invokeLLM(prompt)
        if success:
            score = float(response)
            tag = "_"
        else:
            tag = "E"
        return score, tag

    def gradeNounSentiment(self, text: str, topics: list[str]) -> tuple[dict, str]:
        """
        Parse the sentiment of the review text using the LLM model.
        """
        output: dict = {}
        returnState: str = ""
        prompt: str = textwrap.dedent(f"""
                Your task is to
                1) search list of topics in a review text
                2) calculate sentiment of each topic in text as score
                    -1 (negative), 0 (neutral/not found), 1 (positive).
                3) return sentiments as JSON object. ONLY JSON. No comments, explanations, anything else.
                --- 
                TEXT:
                {text}.
                ---
                TOPICS:
                {topics}
                ---""")
        response, success = self.invokeLLM(prompt)
        if success:
            try:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    output = json.loads(match.group(0))
                    returnState = "_"
                else:
                    output = {}
                    returnState = "E"
            except json.JSONDecodeError:
                returnState = "J"
        else:
            returnState = "E"
        return output, returnState
