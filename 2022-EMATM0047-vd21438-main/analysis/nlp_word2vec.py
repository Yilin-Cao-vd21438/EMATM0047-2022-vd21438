import warnings

warnings.filterwarnings("ignore")
from nlp_utils import *
from nlp_utils import create_ngrams_detectors
import numpy as np
from tensorflow.keras import models, layers, preprocessing
from sklearn import metrics
import json
from IPython.core.display import display, HTML

# Manual NN, for illusration only
def attention_layer(x, neurons):
    Q, K, V = x, x, x
    K = layers.Permute((2, 1))(K)
    QxK = layers.Dense(neurons, activation="softmax")(K)
    QxK = layers.Permute((2, 1), name="attention")(QxK)
    x = layers.multiply([V, QxK])
    return x

'''
NLP model Train and Evaluation
:parameter
    :param dtf_train: array of sequence
    :param y_train: array of classes
    :param dtf_test: array of sequence
	:param y_test: array of classes
	:param verbose: 'auto', 0, 1, or 2
	:param outputDir: output directory
	:param resultFile: output result file
:return
    result of prediction and evaluation in outputDir and resultFile
'''

def evaluate(dtf_train, dtf_test, y_train, y_test, verbose, outputDir, resultFile):
    # The parameters are nlikely slices and we get warning modifying if we do not make copy
    dtf_train = dtf_train.copy()
    dtf_test = dtf_test.copy()

    # not necessary, but useful
    print("-----Ngram Detection-----")
    lst_common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    lst_ngrams_detectors, dtf_ngrams = create_ngrams_detectors(corpus=dtf_train["text_clean"], outputDir=outputDir,
                                                               lst_common_terms=lst_common_terms, min_count=5,
                                                               top=10, figsize=(10, 7))
    ## In order to know the position of a certain word, we can look it up in the vocabulary:
    if (verbose):
        txt = "river bank"
        lst_ngrams_detectors[1][txt.split()]
        print(dtf_ngrams.sample(5))

    '''
    Transforms the corpus into an array of sequences of idx (tokenizer) with same length (padding/truncation)
    Create input for lstm (sequences of tokens)
    '''
    # BOW with keras to get text2tokens without creating the sparse matrix
    corpus = dtf_train["text_clean"]
    lst_corpus = utils_preprocess_ngrams(corpus)

    print("-----Tokenization-----")
    tokenizer = kprocessing.text.Tokenizer(num_words=None, lower=True, split=' ', char_level=False, oov_token=None,
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(lst_corpus)
    dic_vocabulary = tokenizer.word_index
    resultFile.writeln(" ")
    resultFile.writeln(str(len(dic_vocabulary)) + " words")
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    # Padding sequence
    print("-----Padding to sequence-----")
    X = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=50, padding="post", truncating="post")
    resultFile.writeln(" ")
    resultFile.writeln(str(X.shape[0]) + " sequences of length " + str(X.shape[1]))
    resultFile.writeln(" ")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(X == 0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sequences Overview')
    plt.savefig(os.path.join(outputDir, "sequence_overview.png"))
    plt.close(fig)  # disables displaying

    X_train = X

    # Preprocess Test with the same tokenizer
    corpusT = dtf_test["text_clean"]
    lst_corpusT = utils_preprocess_ngrams(corpusT)
    lst_text2seqT = tokenizer.texts_to_sequences(lst_corpusT)
    XT = kprocessing.sequence.pad_sequences(lst_text2seqT, maxlen=X_train.shape[1], padding="post", truncating="post")
    X_test = XT

    '''
    Train Word2Vec from scratch
    Fits Word2Vec model from gensim
    Return nlp model
    '''
    print("-----Constructing word2vec nlp model-----")
    avg_len = np.max([len(text.split()) for text in dtf_train["text_clean"]]) / 2
    lst_corpus, nlp = fit_w2v(corpus=dtf_train["text_clean"], lst_ngrams_detectors=lst_ngrams_detectors,
                              min_count=1, size=300, window=avg_len, sg=1, epochs=30)

    embeddings = vocabulary_embeddings(dic_vocabulary, nlp)

    # Embedding network with Bi-LSTM and Attention layers
    x_in = layers.Input(shape=(X_train.shape[1],))

    # embedding
    x = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings],
                         input_length=X_train.shape[1], trainable=False)(x_in)
    # attention
    # x = attention_layer(x, neurons=X_train.shape[1])  #<-- tensorflow 1 (manual function)
    x = layers.Attention()([x, x])  # <-- tensorflow 2 (included in keras)

    # 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2))(x)

    # final dense layers
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(units=len(np.unique(y_train)), activation='softmax')(x)

    # compile
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # this takes a while
    print("-----Train and Test-----")

    encode_y = True
    epochs = 10
    batch_size = 256
    verbose = 0 if epochs > 1 else 1

    dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
    inverse_dic = {v: k for k, v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])

    # train
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose,
                         validation_split=0.3)

    '''
    Plot loss and metrics of keras training.
    '''
    if epochs > 1:
        metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

        ## training
        ax[0].set(title="Training")
        ax11 = ax[0].twinx()
        ax[0].plot(training.history['loss'], color='black')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
        ax11.legend()

        ## validation
        ax[1].set(title="Validation")
        ax22 = ax[1].twinx()
        ax[1].plot(training.history['val_loss'], color='black')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax22.plot(training.history['val_' + metric], label=metric)
        ax22.set_ylabel("Score", color="steelblue")

        plt.savefig(os.path.join(outputDir, "loss_and_validation.png"))
        plt.close(fig)  # disables displaying

    # test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob] if encode_y is True else [np.argmax(pred)]

    '''
    Evaluate performance of DL model, following metrics are used:

    Accuracy: the fraction of predictions the model got right.
    Confusion Matrix: a summary table that breaks down the number of correct and incorrect predictions by each class.
    ROC: a plot that illustrates the true positive rate against the false positive rate at various threshold settings.
       The area under the curve (AUC) indicates the probability that the classifier will rank a randomly
       chosen positive observation higher than a randomly chosen negative one.
    Precision: the fraction of relevant instances among the retrieved instances.
    Recall: the fraction of the total amount of relevant instances that were actually retrieved.
    '''
    outputDir = "results/word2vec_cnn/arff_20/"
    resultFile = "results/word2vec_cnn/arff_20/results.txt"

    print("-----Start Evaluation-----")
    evaluate_multi_classif(y_test, predicted, predicted_prob, outputDir, resultFile)

    # Select observation
    i = 0
    txt_instance = dtf_test["text"].iloc[i]

    # check true value and predicted value
    f = open(resultFile, "a+")
    f.write("\nTrue: {}--> Pred: {}| Prob: {}".format(y_test[i], predicted[i], round(np.max(predicted_prob[i]), 5)))
