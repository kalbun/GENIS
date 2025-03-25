Overall idea

gaussianiser was written to verify if the text associated to Amazon reviews
allow to build a numeric score more realistic and informative than the star
number.
Due to commercial and psychological pressure, the maximum score of five stars
does not associate with exceptional purchasing experience, but rather to lack
of negative issues (a common behaviour is to remove one star per issue).
Yet, this flattens most of the scores to five stars, making not distinguishable
very different reviews like "best purchase of my life" and "it was ok".

The research hypothesis is that the score distribution should be, in fact,
normal. WHY?????

gaussianiser attempts to recalculate the numeric score by considering the
review text in addition to initial score. To do so, gaussianiser:

- preprocess reviews with nltk tokenizer, keeping only nouns and adjectives
- use embeddings to preserve only words semantically near enough to the
  review overall topic
- clusterise selected words with hdbscan
- stores words belonging to each cluster
- invoke an llm passing the review and cluster words, asking to evaluate
  the sentiment for each of them. For example, if cluster contains
    