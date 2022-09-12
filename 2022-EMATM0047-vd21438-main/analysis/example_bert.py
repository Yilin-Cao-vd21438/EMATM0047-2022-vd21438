from nlp_utils import *
import numpy as np
import transformers
from tensorflow.keras import models, layers, preprocessing
from tensorflow import random
from numpy.random import seed
import tensorflow as tf

seed(1)
random.set_seed(1)
np.random.seed(1)

'''
Pre-trained Bert + TFAutoModelForSequenceClassification + Adam Optimizer
:parameter
    :param X_train: array of sequence
    :param y_train: array of classes
    :param X_test: array of sequence
	:param y_test: array of classes
    :param model: model object - model to fit (before fitting)
    :param encode_y: bool - whether to encode y with a dic_y_mapping
    :param dic_y_mapping: dict - {0:"A", 1:"B", 2:"C"}. If None it calculates
    :param epochs: num - epochs to run
    :param batch_size: num - it does backpropagation every batch, the more the faster but it can use all the memory
:return
    model fitted and predictions
'''


def fit_bert_seqclassif(X_train, y_train, X_test, y_test, encode_y=False, dic_y_mapping=None, model=None, epochs=1,
                     batch_size=2):
    # encode y
    if encode_y is True:
        dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
        y_train = np.array([inverse_dic[y] for y in y_train])
        y_test = np.array([inverse_dic[y] for y in y_test])
    # print(dic_y_mapping)
	
    verbose = 0 if epochs > 1 else 1
	# Correct train test data used
    # print("x_train[0].shape:{}".format(X_train[0].shape))
    # print("x_train[1].shape:{}".format(X_train[1].shape))
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_train),
        y_train
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(X_test),
        y_test
    ))
	
	# Train
    training = model.fit(train_dataset.shuffle(len(train_dataset)).batch(batch_size),
                         epochs=epochs,
                         batch_size=2,
                         shuffle=True,
                         verbose=verbose,
                         validation_data=test_dataset.shuffle(len(test_dataset)).batch(batch_size))
						 
    # Plot loss and metrics, but epoch=1 is enough
	if epochs > 1:
        utils_plot_keras_training(training)

    # Test
    predicted_prob = []
    for i in test_dataset.batch(batch_size):
        logi = model.predict(i).logits
        predicted_prob.extend(logi)
    predicted = [dic_y_mapping[np.argmax(pred)] if encode_y is True else np.argmax(pred) for pred in predicted_prob]
    return training.model, np.array(predicted_prob), predicted


'''
NLP model Train and Evaluation
:parameter
    :param dtf_train: array of sequence
    :param y_train: array of classes
    :param dtf_test: array of sequence
	:param y_test: array of classes
:return
    result of prediction and evaluation in outputDir and resultFile
'''


def evaluate(dtf_train, dtf_test, y_train, y_test):
    seed(1)
    random.set_seed(1)
    np.random.seed(1)
    dtf_train = dtf_train.copy()
    dtf_test = dtf_test.copy()

    y_train = dtf_train["y"].values
    y_test = dtf_test["y"].values

    # Load pre-trained BERT tokenizer (use a lighter version: distil-BERT)
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    nlp = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    # pre-trained bert with config
    config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2, output_hidden_states=False,
                                           output_attentions=True)
    lst_vocabulary = list(tokenizer.vocab.keys())

    seed(1)
    random.set_seed(1)
    np.random.seed(1)

    # check nlp model
    txt = "river bank"
    X = embedding_bert(txt, tokenizer, nlp, log=True)
    print("-----Check NLP model-----")
    print("shape:", X.shape)
    print("mean:", np.mean(X[1]))

    X_train = tokenize_bert(corpus=dtf_train["text"], tokenizer=tokenizer, maxlen=128)
    X_test = tokenize_bert(corpus=dtf_test["text"], tokenizer=tokenizer, maxlen=128)

    # check feature creation
    # i = 0
    # print("-----Check feature creation-----")
    # print("txt: ", dtf_train["text"].iloc[0])
    # print("idx: ", X_train[0][i])
    # print("mask: ", X_train[1][i])
    # print("segment: ", X_train[2][i])
	
    # Pre-trained distil-BERT + fine-tuning (transfer learning)
    model = transformers.TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                              num_labels=len(np.unique(y_train)))
    model.summary()
	# learning rate should be less than 1.0 and greater than 1e-5
	# default value for epsilon = 1e-8
	# https://www.andreaperlato.com/theorypost/the-learning-rate/
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

    seed(1)
    random.set_seed(1)
    np.random.seed(1)

    print("-----Train and Test-----")
    # this takes a while
	# unfortunately my computer could only do batch size 2, it is slow, but will not kill my computer :(
    model, predicted_prob, predicted = fit_bert_seqclassif(X_train, y_train, X_test, y_test, encode_y=True,
                                                        model=model, epochs=1, batch_size=2)
    
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
    outputDir = "results/bert_cnn/arff_10/"
    resultFile = "results/bert_cnn/arff_10/results.txt"
    print("-----Start Evaluation-----")
    evaluate_multi_classif(y_test, predicted, predicted_prob, outputDir, resultFile)
