# from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
# https://github.com/mdipietro09/DataScience_ArtificialIntelligence_Utils

import sys
import nlp_utils

from stopwatch import Stopwatch
from outstream import OutStream

# import nltk
# nltk.download('stopwords')

## for data
import json
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for processing
import re
import nltk

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing

## for explainer
from lime import lime_text

## for word embedding
import gensim
import gensim.downloader as gensim_api

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

## for bert language model
import transformers

## The different approached, tdidf, word embeddings, bert
import nlp_tdidf
import nlp_word2vec
import example_bert

## for decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn import naive_bayes
from sklearn import svm

## for saving results
import os

'''
Cleans text by first removing proper nouns and then calliing nlp_utils.utils_preprocess_text
to clean up the text.  We remove stop words, puctuation, and lemmatize.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
:return
    cleaned text
'''


def clean_text(text, lst_stopwords):
    ## Before cleaning, remove proper nouns (the case is used to determine proper nouns)
    text = nlp_utils.remove_proper_nouns(text)
    # text = utils_preprocess_text( text, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords )
    text = nlp_utils.utils_preprocess_text(text, lst_regex=None, punkt=True, lower=True, lst_stopwords=lst_stopwords,
                                           stemm=False, lemm=True)
    return text


# =======================================================================
# Entry point
# =======================================================================

def main():
    """
    Main entry point for program.  Takes any parameters from command line.
    """
    if (len(sys.argv) < 4):
        print('  Usage: python nlp <JSON filename> <verbose>')
        print('             <feature representaton {tdidf, word2vec, bert}>')
        print('             <classifier {nb, cnn, svm, dt>')
        print('')
        print('             nb=multinomial naive bayes')
        print('             cnn=convolutional neural network')
        print('             svm=support vector machine')
        print('             dt=decision tree')
        print('Example: python nlp arff_5.json true tdidf nb')
        return

    jsondataFilename = sys.argv[1]
    verbose = bool(sys.argv[2])
    featureRepresentation = sys.argv[3]
    classifierName = sys.argv[4]

    # --------------------------------------------------------------------#
    # Each feature representation {tfidf, word2vec} can be paired
    # with a classifier {nb=multinomial naive bayes, cnn=convolutional neural network, svm=support vector machine}
    # --------------------------------------------------------------------#
    classifier = None
    if (classifierName == 'dt'):
        classifier = DecisionTreeClassifier(random_state=1234, criterion='gini', max_depth=6)
        # classifier = DecisionTreeClassifier(random_state=1234)
    elif (classifierName == 'nb'):
        classifier = naive_bayes.MultinomialNB()
    elif (classifierName == 'svm'):
        C = 1.0  # SVM regularization parameter
        # classifier = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, probability=True)
        classifier = svm.SVC(kernel='rbf', gamma=0.7, C=C, probability=True)

    if (featureRepresentation == 'tdidf' and classifier == None):
        print('Classifier unsupported ' + classifierName)
        return

    # --------------------------------------------------------------------#
    # Read the json file into a pandas Dataframe
    # --------------------------------------------------------------------#
    lst_dics = []
    with open(jsondataFilename, mode='r', errors='ignore') as json_file:
        for dic in json_file:
            lst_dics.append(json.loads(dic))  ## print the first one
    # if( verbose ):print(lst_dics[0])

    # --------------------------------------------------------------------#
    # Get the arrf filename without the extension or path,
    # use as name for results directory
    # --------------------------------------------------------------------#
    # results
    #     tdidf
    #         arff_1
    #         arff_2
    #     word2vec
    #         arff_1
    #         arff_2
    resultsDir = 'results'
    approach = featureRepresentation + '_' + classifierName
    approachDir = os.path.join(resultsDir, approach)
    baseFilename = os.path.splitext(os.path.basename(jsondataFilename))[0]
    sentencesPerInstance = int(baseFilename.split('_')[1])
    print('sentencesPerInstance=' + str(sentencesPerInstance))
    outputDir = os.path.join(approachDir, baseFilename)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Create file to write out results to
    resultPath = os.path.join(outputDir, 'results.txt')
    resultFile = OutStream(resultPath)
    resultFile.writeln('featureRepresentation=' + featureRepresentation)
    resultFile.writeln('classifier=' + classifierName)

    ## The original dataset contains over 30 categories, but for the purposes of this tutorial,
    ## I will work with a subset of 3: Entertainment, Politics, and Tech.

    ## create dtf
    dtf = pd.DataFrame(lst_dics)  ## filter categories

    # --------------------------------------------------------------------#
    # Undersample to avoid class imbalance
    # --------------------------------------------------------------------#
    dtf.info(verbose=True)
    # print( dtf.describe(include='all') )
    groupedDataframe = dtf.groupby('category', group_keys=False).count()
    print(groupedDataframe)
    resultFile.writeln(groupedDataframe)

    # iterating dataframe, row index and row values in dictionary
    minCount = None
    for category, row in groupedDataframe.iterrows():
        categoryCount = row['headline']
        if (minCount is None or categoryCount < minCount):
            minCount = categoryCount
    sampleSize = int(minCount * 0.50)  # leave some randomness even in the minority class
    print("sampleSize=" + str(sampleSize))
    # return

    # Originally filtered to only 3 of 7 topics, we ignore, only have 2 topics [digital, print]
    if (jsondataFilename == "data.json"):
        dtf = dtf[dtf["category"].isin(['ENTERTAINMENT', 'POLITICS', 'TECH'])][["category", "headline"]]
    else:
        ## Computational Linguistics requires 30 GB!  We need to sample, actually need to stratify sample
        ## Need to sample n or at least all of the them by class, this is stratified sampling
        ## Default sample is without replacement.  n=40000 is good compromise, 10k per class after sampling
        # Sampling -> comment out to see class imbalance
        dtf = dtf.groupby('category', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sampleSize)))  # select 100 from both
        # dtf = dtf[ dtf["category"].isin(['print','digital']) ][["category","headline"]]

    # dtf = dtf.sample( frac=0.25 )

    # --------------------------------------------------------------------#
    ## rename columns
    # --------------------------------------------------------------------#
    dtf = dtf.rename(columns={"category": "y", "headline": "text"})
    ## print 5 random rows
    print(dtf.sample(5))

    ## In order to understand the composition of the dataset, I am going
    ## to look into the univariate distribution of the target by showing
    ## labels frequency with a bar plot.

    ## Turn interactive mode off for all plots???
    # plt.ioff()

    '''
    fig, ax = plt.subplots()
    fig.suptitle("y", fontsize=12)
    dtf["y"].reset_index().groupby("y").count().sort_values(by=
                                                            "index").plot(kind="barh", legend=False,
                                                                          ax=ax).grid(axis='x')
    plt.savefig(os.path.join(outputDir, "class_count.png"))
    plt.close(fig)  # disables displaying
    '''

    # --------------------------------------------------------------------#
    ## That function removes a set of words from the corpus if given.
    ## I can create a list of generic stop words for the English vocabulary
    ## with nltk (we could edit this list by adding or removing words).
    lst_stopwords = nltk.corpus.stopwords.words("english")
    # if( verbose ): print( lst_stopwords )

    ## Now I shall apply the function I wrote on the whole dataset and
    ## store the result in a new column named “text_clean” so that you
    ## can choose to work with the raw corpus or the preprocessed text.
    # dtf["text_clean"] = dtf["text"].apply(lambda x:
    #      utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,
    #      lst_stopwords=lst_stopwords))
    # --------------------------------------------------------------------#
    dtf["text_clean"] = dtf["text"].apply(lambda text: clean_text(text, lst_stopwords))

    print("### DISTRIBUTIONS ###")
    print(dtf.head())
    # Stats on processed data
    dtf = nlp_utils.add_text_length(dtf, "text_clean")
    # Stats on raw data
    # dtf = nlp_utils.add_text_length(dtf, "text")
    print(dtf.head())
    nlp_utils.plot_distributions(dtf, x="word_count", y="y", bins=20, figsize=(15, 5),
                                 outputFile=os.path.join(outputDir, "word_count.png"))
    nlp_utils.plot_distributions(dtf, x="avg_word_length", y="y", bins=20, figsize=(15, 5),
                                 outputFile=os.path.join(outputDir, "avg_word_length.png"))
    nlp_utils.plot_distributions(dtf, x="sentence_count", y="y", bins=10, figsize=(15, 5),
                                 outputFile=os.path.join(outputDir, "sentence_count.png"))
    nlp_utils.plot_distributions(dtf, x="avg_sentence_length", y="y", bins=20, figsize=(15, 5),
                                 outputFile=os.path.join(outputDir, "avg_sentence_length.png"))
    # return

    ## 70/30 train/test split
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)  ## get target
    y_train = dtf_train["y"].values
    y_test = dtf_test["y"].values

    # --------------------------------------------------------------------#
    # Now that the data has been prepared, evaluate using the different approaches
    # --------------------------------------------------------------------#
    clock = Stopwatch()  # keep track of how long the classification time takes

    if (featureRepresentation == "tdidf"):
        nlp_tdidf.evaluate(dtf_train, dtf_test, verbose, outputDir, resultFile, classifier)
    elif (featureRepresentation == "word2vec"):
        nlp_word2vec.evaluate(dtf_train, dtf_test, y_train, y_test, verbose, outputDir, resultFile)
    elif (featureRepresentation == "bert"):
        example_bert.evaluate(dtf_train, dtf_test, y_train, y_test)

    time = clock.elapsedTime()
    timeStr = 'classification_processing_time=' + str(time)
    print(timeStr)
    resultFile.writeln(timeStr)
    resultFile.close()


# =======================================================================
# Command line bootstrap
# =======================================================================
if __name__ == '__main__':
    main()
