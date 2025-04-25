Valid up to version 0.5.0

# CHANGELOG

0.5.0: added the ability to generate random corrected scores
0.4.0: added new run parameters

# OVERALL IDEA

gaussianiser was written to verify if the text associated to Amazon reviews
allow to build a numeric score more realistic and informative than the star
number.
Due to commercial and psychological pressure, the maximum score of five stars
does not associate with exceptional purchasing experience, but rather to lack
of negative issues (a common behaviour is to remove one star per issue).
Yet, this flattens most of the scores to five stars, making not distinguishable
very different reviews like "best purchase of my life" and "it was ok".

The research hypothesis is that the score distribution should be, in fact,
normal. (WHY?????)

gaussianiser attempts to recalculate the numeric score by considering the
review text in addition to initial score. To do so, gaussianiser:

- preprocess reviews with nltk tokenizer, keeping only nouns and adjectives
- use embeddings to preserve only words semantically near enough to the
  review overall topic
- clusterises selected words with hdbscan
- stores words belonging to each cluster
- assigns each review to one and only one cluster
- invoke an llm passing the review and cluster words, asking to evaluate
  the sentiment for each of them. The prompt instructs the llm to assign
  a numerical score from -1 to +1 with a resolution of 0.5.
  For example, if cluster contains _reader_, _reading_ and _readership_
  while the review is _I read this book with great pleasure_, then the
  llm may return a score of 0.5 or 1.
- calculates a correction score as the sum of all the words found (0 if
  no words were found).
- calculates the final corrected score as the original score plus the
  correction multiplied by a factor:

    S_o = original score
    C = correction
    a = scale factor
    S_c = corrected score

    S_c = S_o + C * a
    
# RUN GAUSSIANISER

## Create the environment

To run gaussianiser, create a new python environment and install there the
python interpreter version 3.12.
For example, with miniconda, execute:

    conda create -n myenv
    conda activate myenv
    conda install python=3.12

Once python is installed, invoke pip to install the requirements:

    pip install -r requirements.txt

The operation may take several minutes, but it should run with no problems.

## Create key file

gaussianiser needs access to Mistral APIs. To do so, you must create a text
file name key.py and write your API key in this format:

    MistraAIKey: str = <your key>

You can use different LLMs, but in this case you will need to manually modify
the code in sentiments.py.

## Run parameters

Gaussianiser has several parameters that influence how it works.
To get acquainted with them, you can add the -h or --help switch and
read the resulting message.

filename
  the only mandatory parameter, defines a Jsonl file containing the
  reviews to analyse.
  
-s SEED
  Allows to set the seed for the random generator. The default is 1967.
  Using a fixed seed allows to get repeatable results.

-m MAX_REVIEWS
  Limits the number of analysed reviews. Default is 1000.

-r RUNS
  Allows to repeat the analysis multiple times. At each run gaussianiser
  selects a different subset of reviews and repeat the clustering and
  sentiment analysis. Data from all runs are collected into a single file,
  see later "output files".
  Default is one run.

-n --no_images
  By default, gaussianiser shows the results of score reranking in a chart.
  You can prevent the chart generation with this flag. It could be useful
  in case you only need the result files, or when you set RUNS to a high
  values and want to make the process automatic.

-hc --hcluster
  Allows to modify the minimum cluster size used in HDBSCAN. Default 3.

-hs --hsize
  Allows to modify the minimum sample used in HDBSCAN. Default 2.

-t --threshold
  Allows to modify the relevance threshold for embeddings. A word is
  classified as relevant if the semantic similarity is at least this value.
  Default 0.25.

-b --bypass
  Bypasses embeddings and sentiment caches. This is useful to run analysis
  that changes the relevance threshold or HDBSCAN parameters.

-e --earlystop
  Stop after printing cluster table. Do not execute LLM sentiment analysis.

-fr --forcerandom
  If this flag is specified, sentiment scores for each topic are not
  calculated with LLM but randomly selected. This is useful for statistical
  purposes when attempting to demonstrate null hypothesis.

## Where to find reviews?

  There are many public repositories with Amazon review, but one of the best
  can be found here:
  
  https://amazon-reviews-2023.github.io/

  This site hosts an impressive collection of reviews for various categories.
  To use them, just pick a category, click the <review> link, gunzip the
  downloaded file and put it in gaussianiser directory.
  Be careful, many files are really huge!

  You can also use different sources, provided the file is in jsonl format
  and contain, as a minimum, the review text and the score in stars (1-5).

## First run and automatic downloads

nltk and sentence_transformer libraries need additional packets to work.
During the first run, they will connect to the internet and automatically
download additional files. The operation is long but it takes place once.

# DATA STORAGE

Gaussianiser saves a number of files while running and put them in different
directories to help keeping data in order.
Moreover, it also caches data to reduce recalculation to a minimum in case of multiple runs.

## Cache files

If the topic file is called **topic**.jsonl, then Gaussianiser creates these cache files under subdirectory **topic**:

- **topic**_embeddings_cache.pkl  
  A dictionary that associates a word with its embeddings. Stored in pickle format.

- **topic**_spellcheck_cache.json  
  A dictionary associating each original sentence with its spellchecked version. Caching the spellcheck result reduces the processing time for this intensive operation.

- **topic**_correction_cache.json  
  A dictionary that associates each word with its corrected form. The file contains only words that were corrected. This cache minimizes the repeated work of spell checking.

- **topic**_preprocessing_cache.json  
  A dictionary where each corrected sentence is associated with tokens extracted during preprocessing (tokenized nouns) and the original score.

- **topic**_sentiment_cache.jsonl  
  A dictionary where each corrected sentence is associated with the terms to use for sentiment analysis and the resulting score.

- **topic**_results.csv  
  Contains the original and reranked scores along with additional details.

Gaussianiser also creates a sub-subdirectory named after the seed value and stores there the cache of LLM-based sentiment analysis, for example:

  **magazines**_sentiments_cache.jsonl

We recommend not to modify these files.

## The results csv file

This file contains the original and reranked scores, along with other details.
The first row contains the data header and should be rather self-explanantory.

  timestamp,run,original,adjusted,seed,reviewID,sentence,appVersion

reviewID refers to the position of the review in the file. The first review is
ID 0, the second has ID 1, and so on.

run is the run number, starting from 1. For example, if you set -r 10 when
invoking gaussianiser, you will find that the field <run> moves from 1 to 10.
Yet, by calling gaussianiser ten times, you will get all the data with <run>
set to 1. Also, considering that the seed is predetermined, all the ten runs
will exhibit exactly the same parameters.
