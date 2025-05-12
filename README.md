Valid up to version 0.10.0

# CHANGELOG

0.10.0: Switched from notebook to regular python file. Refactoring of many parts. No more sentiment cache, everything is now in the preprocessing cache. Involvement of LLM to give a score to the review.

# OVERALL IDEA

GENIS was written to verify if the text associated to Amazon reviews allow to build a numeric score more realistic and informative than the star number. Due to commercial and psychological pressure, the maximum score of five stars does not associate with exceptional purchasing experience, but rather to lack of negative issues (a common behaviour is to remove one star per issue). Yet, this flattens most of the scores to five stars, making not distinguishable very different reviews like "best purchase of my life" and "it was ok".

GENIS attempts to reconstruct the numeric score by considering the review text in addition to initial score. GENIS uses a combination of traditional NLP and LLM to calculate a score from 1 to 10.

GENIS uses NLP techniques to extract sentiment-loaded pairs of nouns and adjectives (like "good music" or "yellow duck"). These context-unaware data are prefiltered by using VADER algorithm. The above-threshold nouns are then passed to a LLM along with the review, which attributes a sentiment (-1, 0 or +1) to each noun. By summing up separately the negative, neutral and positive sentiments, we obtain the three final scores of GENIS. To convert them into a number from 1 to 10, we trained a random forest regressor that gets in input the three GENIS scores and the original rating in stars.

Preliminary tests suggest that GENIS performs better than the LLM and VADER separately when comparing calculated scores with human-made grades.
Moreover, the nouns extracted from the review account for a better explainability of the score: the GENIS methodology not only grades the reviews, but can also explain which specific aspects contributed.

GENIS performs these steps on a review:
- cleanup, unescaping and other regularisation operations.
- extraction of noun-adjective pairs from the review, using Spacy NLP. We consider both adjectival modifiers (e.g., "good" in "good music") and adjectival complements (e.g., "good" in "the product is good"). In this version, we don't consider other sentiment-loaded pairs like verb-noun ("I love this music"). This could be considered in next works.
- Pairs filtering with VADER, keeping only those with absolute compound value at least 0.05. Note that the selection is context-independent.
- Sentiment calculation with LLM, which receives the review and the nouns from the filtered pairs (not the adjectives). The LLM is asked to give a value of -1 (negative), 0 (neutral) or +1 (positive) to each noun. The query uses zero-shot with a few examples embedded.
- Separate count of positive, neutral and negative aspects, which represent the GENIS scores.
- training or inference of a random forest classifier.

# GENISCALC vs GENISTRAIN

GENIS software is made of two distinct modules:

- genisCalc.py, used to preprocess reviews and generate data for human grading of reviews and training of the ML model
- genisTrain.py that get data obtained from genisCalc and use them to train a classifier.

# RUN GENIS

## Create the environment

To run GENIS scripts, it is recommended to create a new python environment and install there the python interpreter version 3.12.
For example, with miniconda, execute:

    conda create -n GENIS
    conda activate GENIS
    conda install python=3.12

## install dependencies

Once python is installed, invoke pip to install the requirements:

    pip install -r requirements.txt

The operation may take several minutes, but it should run with no problems.

Conda recommends to first try to install from their distribution channel (conda install) and use pip only for packets not found.

## download Spacy model

The last phase is loading Spacy data. From the command line and inside the correct environment, invoke:

    python -m spacy download en_core_web_sm


## run genisCalc

### Create key file

genisCalc needs access to Mistral APIs. To do so, you must create a text file name key.py and write your API key in this format:

    MistraAIKey: str = <your key>

*Please note*: to shorten LLM interaction, we run queries in parallel. The number of queries varies with the Mistral license you have: for paid licenses, at current date (May 2025) you have a limit of six and of two million tokens per minute.
You can change the number of concurrent queries by modifying the semaphore init in sentiments.py:

    self.llmSemaphore = threading.Semaphore(6) <-- change it as needed

You can use different LLMs, but you will need to manually modify the code in sentiments.py.

### command line parameters

genisCalc has several parameters that influence how it works.
To get acquainted with them, you can add the -h or --help switch and
read the resulting message.

filename
  the only mandatory parameter, defines a Jsonl file containing the
  reviews to analyse. The extension is optional: rubberducks and rubberducks.jsonl have the same effect.
  
-s SEED
  Allows to set the seed for the random generator. The default is 1967.
  Using a fixed seed allows to get repeatable results.

-m MAX_REVIEWS
  Limits the number of analysed reviews. Default is 1000.

### An example

You got a file of great reviews called redapples.jsonl. Then you call:

    python.exe redapples

This will process 1000 reviews with random seed 1967, producing various files described in "genisCalc files".

### First run and automatic downloads

nltk and sentence_transformer libraries need additional packets to work.
During the first run, they will connect to the internet and automatically
download additional files. The operation is long but it takes place once.

### genisCalc output files

genisCalc stores all the files into subdirs of a directory called data.

If the topic file is called **topic**.jsonl and the random seed is **seed**, then genisCalc creates:

    data/**topic**
    data/**topic**/**seed**

File created or updated are:

- data/**topic**/correction_cache.json  
  A dictionary that associates each word with its corrected form. The file contains only words that were corrected. This cache minimizes the repeated work of spell checking.

- data/**topic**/preprocessing_cache.json
  A dictionary storing all the aspects related to a review, like associated pairs, VADER and LLM score, cleaned text. Don't change this file unless you know very well what you are doing.

- data/**topic**/**seed**/selected_reviews.csv
  This file contains the reviews to manually grade. The human reviewer should read the texts in column 1 and replace the scores in column 2.
  Note that <b>this file is replaced at every run</b>! This could be quite unpleasant if you already annotated manual grades. To avoid unintended overwrites, genisCalc asks for a confirmation before proceeding.

## You want to stay in the loop? Then it's your turn!

After genisCalc has successfully completed the run, and before you invoke genisTrain, you need to manually grade a selection of 100 reviews. The file is in CSV format and its directory depends on the name of your jsonl review file and the seed. For example, if you launched:

    python.exe genisCalc.py shamansticks.json -s 4321

then the file will be located in:

    data/shamansticks/4321/selected_reviews.csv

Open the file with an appropriate application (Excel for example, or a notepad of your choice). You will see three columns: the first is the review in readable form, the second contains the scores, the third is the original review.

Please attribute a grade to each review. Put yourself in customer's clothes, try to summarize the text in a single number. Not easy but we humans are good at this!
Please also resist the temptation to use an LLM because this may taint the results of the training.

## run genisTrain

Once the manual grading is completed, you can invoke genisTrain. This will:
- train the random forest classifier
- save the classifier in a simple pickle file for later usage
- get preliminary metrics.

genisTrain receives two parameters:

List of space-separated directories:
  names of directories where genisCalc stored its data. For example, if you processed rubberducks.jsonl and lightsabers.json, then invoke genisTrain this way:

    python.exe genisTrain.py rubberducks lightsabers

  You can specify as many directories as you want.

-s SEED:
  Allows to set the seed for the random generator. The default is 1967.
  Specify the same seed used for genisCalc.

### genisTrain output files

genisTrain will produce two files:

    data/overall_results containing the consolidated data in csv (see DATA STORAGE)
    data/random_forest_classifier.pkl, the serialized dump of the classifier in pickle format.

# Where to find reviews?

  There are many public repositories with Amazon review, but one of the best can be found here:
  
  https://amazon-reviews-2023.github.io/

  This site hosts an impressive collection of reviews for various categories. To use them, just pick a category, click the <review> link, gunzip the downloaded file and put it in GENIS directory. The .jsonl provided are already in the right format.
  Be careful, many files are really huge!

  You can also use different sources, provided the file is in jsonl format and contain, as a minimum, the review text and the score in stars (1-5) in two columns named respectively "text" and "rating". Should you column have a different name, you need to change the code in genisCalc.py

    label_text: str = "text"
    label_rating: str = "rating"  <-- change labels as needed

