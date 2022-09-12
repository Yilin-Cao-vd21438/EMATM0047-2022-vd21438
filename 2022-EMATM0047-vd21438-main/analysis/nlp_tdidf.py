## for data
# import json
import pandas as pd
import numpy as np

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, \
    feature_selection, metrics

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for explainer
from lime import lime_text

## for decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# from dtreeviz.trees import dtreeviz # remember to load the package
from sklearn import preprocessing

## for saving results
import os
from outstream import OutStream
from sklearn import svm

'''
Converts unique categories into unique, positive integers.
Efectively replacement for preprocessing.LabelEncoder()
'''


def transform(y, nameToIntegerDict):
    # dtf_train['y'] = dtf_train['y'].apply( lambda y: transform(y, nameToIntegerDict) )
    if (len(nameToIntegerDict) == 0):
        nameToIntegerDict[y] = 0

    if (y not in nameToIntegerDict):
        values = nameToIntegerDict.values()
        maxInteger = max(values)
        nameToIntegerDict[y] = maxInteger + 1

    index = nameToIntegerDict[y]
    return str(index)  # so that the column is of type object instead of int64


'''
Trains classifier using dtf_train and evaluates using dtf_test.
In particular, transforms text into bag of words and then weights
the words using TD-IDF. The weighted words are then trained using 
a NaiveBayes classifier.
:parameter
    :param dtf_train: dataframe - dtf with a text and category column
    :param dtf_test: dataframe - dtf with a text and category column
    :param verbose: boolean - true to show details of processing
    :param outputDir: path - place to store matplotlib images and text results
    :param decisionTree: bool - True to use decision tree classifier, False to use NaiveBayes classifier
:return
    None
'''


def evaluate(dtf_train, dtf_test, verbose, outputDir, resultFile, classifier):
    # The parameters are nlikely slices and we get warning modifying if we do not make copy
    dtf_train = dtf_train.copy()
    dtf_test = dtf_test.copy()

    ## For decision tree the y values need to be integer lookup
    if (isinstance(classifier, DecisionTreeClassifier)):
        print("dtf_train before")
        print(dtf_train.dtypes)
        print(dtf_train)
        nameToIntegerDict = dict()  # only used when decision tree
        dtf_train['y'] = dtf_train['y'].apply(lambda y: transform(y, nameToIntegerDict)).astype('str')
        dtf_test['y'] = dtf_test['y'].apply(lambda y: transform(y, nameToIntegerDict)).astype('str')
        print('Labels')
        print(nameToIntegerDict)
        print("dtf_train after")
        print(dtf_train.dtypes)
        print(dtf_train)

    y_train = dtf_train["y"].values
    y_test = dtf_test["y"].values

    ## Count (classic BoW), not really needed, overwritten anyways
    vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1, 2))

    ## Tf-Idf (advanced variant of BoW)
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    ## Now I will use the vectorizer on the preprocessed corpus of the
    ## train set to extract a vocabulary and create the feature matrix.
    corpus = dtf_train["text_clean"]
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    ## The feature matrix X_train has a shape of
    ## 34,265 (Number of documents in training) x 10,000 (Length of vocabulary)
    ## and it’s pretty sparse:

    # sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')
    # plt.show()

    ## In order to know the position of a certain word, we can look it up in the vocabulary:
    if (verbose and "new york" in dic_vocabulary):
        word = "new york"
        print(dic_vocabulary[word])

    ## If the word exists in the vocabulary, this command prints a number N,
    ## meaning that the Nth feature of the matrix is that word.

    ## In order to drop some columns and reduce the matrix dimensionality,
    ## we can carry out some Feature Selection, the process of selecting a
    ## subset of relevant variables. I will proceed as follows:

    ## 1. treat each category as binary (for example, the “Tech” category is 1 for the Tech news and 0 for the others);
    ## 2. perform a Chi-Square test to determine whether a feature and the (binary) target are independent;
    ## 3. keep only the features with a certain p-value from the Chi-Square test.

    ## Feature reduction
    y = dtf_train["y"]
    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y == cat)
        dtf_features = dtf_features.append(pd.DataFrame({"feature": X_names, "score": 1 - p, "y": cat}))
        dtf_features = dtf_features.sort_values(["y", "score"], ascending=[True, False])
        dtf_features = dtf_features[dtf_features["score"] > p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()

    ## I reduced the number of features from 10,000 to 3,152 by keeping the most statistically relevant ones.
    ## Let’s print some:
    for cat in np.unique(y):
        resultFile.writeln("# {}:".format(cat))
        resultFile.writeln("  . selected features: {}".format(len(dtf_features[dtf_features["y"] == cat])))
        resultFile.writeln(
            "  . top features:" + str(",".join(dtf_features[dtf_features["y"] == cat]["feature"].values[:10])))
        resultFile.writeln(" ")

    ## We can refit the vectorizer on the corpus by giving this new set of words as input.
    ## That will produce a smaller feature matrix and a shorter vocabulary.

    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_

    ## The new feature matrix X_train has a shape of is
    ## 34,265 (Number of documents in training) x 3,152 (Length of the given vocabulary)
    ## Let’s see if the matrix is less sparse:

    ## It’s time to train a machine learning model and test it.
    ## I recommend using a Naive Bayes algorithm: a probabilistic classifier that
    ## makes use of Bayes’ Theorem, a rule that uses probability to make predictions
    ## based on prior knowledge of conditions that might be related.
    ## This algorithm is the most suitable for such large dataset as it considers
    ## each feature independently, calculates the probability of each category, and
    ## then predicts the category with the highest probability.
    # classifier = naive_bayes.MultinomialNB()

    ## I’m going to train this classifier on the feature matrix and then test
    ## it on the transformed test set. To that end, I need to build a scikit-learn pipeline:
    ##   a sequential application of a list of transformations and a final estimator.
    ## Putting the Tf-Idf vectorizer and the Naive Bayes classifier in a pipeline allows
    ## us to transform and predict test data in just one step.

    ## pipeline
    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])  ## train classifier

    model["classifier"].fit(X_train, y_train)  ## test
    X_test = dtf_test["text_clean"].values
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)

    ## Visualisations specific to decision trees
    ## see https://mljar.com/blog/visualize-decision-tree/
    ## see https://towardsdatascience.com/how-to-prune-decision-trees-to-make-the-most-out-of-them-3733bd425072
    if (isinstance(classifier, DecisionTreeClassifier)):
        ## Textual decison tree
        text_representation = tree.export_text(classifier, feature_names=X_names)
        resultFile.writeln(text_representation)

        ## Not a very nice grapical decision tree
        # fig = plt.figure(figsize=(25,20))
        # tree.plot_tree(classifier, feature_names=X_names, class_names=np.unique(y), filled=True, fontsize=8)

        ## Nice graphical decision tree but cannot get to work
        # Y_names = list(np.unique(y_train))
        # print(Y_names)
        # print(y_train)
        # viz = dtreeviz(classifier, X_train, y_train,
        #        target_name="category",
        #        feature_names=X_names,
        #        class_names=Y_names )
        # viz.view()

    ## We can now evaluate the performance of the Bag-of-Words model, I will use the following metrics:

    ## Accuracy: the fraction of predictions the model got right.
    ## Confusion Matrix: a summary table that breaks down the number of correct and incorrect predictions by each class.
    ## ROC: a plot that illustrates the true positive rate against the false positive rate at various threshold settings.
    ##    The area under the curve (AUC) indicates the probability that the classifier will rank a randomly
    ##    chosen positive observation higher than a randomly chosen negative one.
    ## Precision: the fraction of relevant instances among the retrieved instances.
    ## Recall: the fraction of the total amount of relevant instances that were actually retrieved.

    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)

    # Shitty Python API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    # Apparently two class (i.e. binary) must be handled differently than three or more classses (why, because scikit-learn sucks)
    auc = None
    if (len(classes) > 2):
        auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    elif (len(classes) == 2):
        auc = metrics.roc_auc_score(y_test, predicted_prob[:, 1], multi_class="ovr")

    # ignore verbosity, we need to see these
    resultFile.writeln("Accuracy:{}".format(round(accuracy, 2)))
    resultFile.writeln("Auc:{}".format(round(auc, 2)))
    resultFile.writeln("Detail:")
    resultFile.writeln(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    plt.savefig(os.path.join(outputDir, "cm.png"))
    plt.close(fig)  # disables displaying

    # I think this is with the roc and not the cm???
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i], predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(recall, precision))
                   )
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)

    plt.savefig(os.path.join(outputDir, "roc.png"))
    plt.close(fig)  # disables displaying

    ## The BoW model got 85% of the test set right (Accuracy is 0.85),
    ## but struggles to recognize Tech news (only 252 predicted correctly).

    ## Let’s try to understand why the model classifies news with a certain
    ## category and assess the explainability of these predictions.
    ## The lime package can help us to build an explainer.
    ## To give an illustration, I will take a random observation from the
    ## test set and see what the model predicts and why.

    ## select observation
    i = 0
    txt_instance = dtf_test["text"].iloc[i]

    ## check true value and predicted value
    resultFile.writeln(
        "True:{}--> Pred:{}| Prob:{}".format(y_test[i], predicted[i], round(np.max(predicted_prob[i]), 5)))

    ## show explanation
    explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
    explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=10)
    # explained.show_in_notebook(text=txt_instance, predict_proba=False)

    explained.save_to_file(os.path.join(outputDir, 'tdidf-explained.html'), text=txt_instance, predict_proba=False)
