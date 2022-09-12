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
Pre-trained Bert + Fine-tuning (transfer learning) with tf2 and transformers.
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

def fit_bert_classif(X_train, y_train, X_test, y_test, encode_y=False, dic_y_mapping=None, model=None, epochs=1, batch_size=2):
    # encode y
    if encode_y is True:
        dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
        y_train = np.array([inverse_dic[y] for y in y_train])
        y_test = np.array([inverse_dic[y] for y in y_test])
    # print(dic_y_mapping)

    # Model
    if model is None:
        # inputs
        idx = layers.Input((128), dtype="int32", name="input_ids")
        masks = layers.Input((128), dtype="int32", name="attention_mask")
        # segments = layers.Input((128), dtype="int32", name="input_segments")
        # pre-trained distil-bert
        bert = transformers.TFBertModel.from_pretrained("distilbert-base-uncased")
        bert_out = bert([idx, masks])
        # fine-tuning
        x = layers.GlobalAveragePooling1D()(bert_out[0])
        x = layers.Dense(64, activation="relu")(x)
        y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)
        # compile
        model = models.Model([idx, masks], y_out)
        for layer in model.layers[:4]:
            layer.trainable = False
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

    # Train
    verbose = 0 if epochs > 1 else 1

    training = model.fit(x=[np.array(X_train['input_ids']), np.array(X_train['attention_mask'])],
          y = y_train,
          epochs=epochs,
          shuffle=True,
          batch_size=batch_size,
          verbose=verbose,
          validation_data=([np.array(X_test['input_ids']), np.array(X_test['attention_mask'])], y_test))
          
    if epochs > 1:
        utils_plot_keras_training(training)

    # Test
    predicted_prob =  model.predict([np.array(X_test['input_ids']), np.array(X_test['attention_mask'])])
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


def evaluate(dtf_train, dtf_test, y_train, y_test, outputDir):
    seed(1)
    random.set_seed(1)
    np.random.seed(1)
    dtf_train = dtf_train.copy()
    dtf_test = dtf_test.copy()

    y_train = dtf_train["y"].values
    y_test = dtf_test["y"].values

    # Load pre-trained BERT tokenizer (use a lighter version: distil-BERT)
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    # pre-trained bert with config
    config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2, output_hidden_states=False,
                                           output_attentions=True)
    lst_vocabulary = list(tokenizer.vocab.keys())

    seed(1)
    random.set_seed(1)
    np.random.seed(1)

    # # check nlp model
    # txt = "river bank"
    # X = embedding_bert(txt, tokenizer, nlp, log=True)
    # print("-----Check NLP model-----")
    # print("shape:", X.shape)
    # print("mean:", np.mean(X[1]))
	
	# Tokenize
    X_train = tokenize_bert(corpus=dtf_train["text"], tokenizer=tokenizer, maxlen=128)
    X_test = tokenize_bert(corpus=dtf_test["text"], tokenizer=tokenizer, maxlen=128)

    # check feature creation
    # i = 0
    # print("-----Check feature creation-----")
    # print("txt: ", dtf_train["text"].iloc[0])
    # print("tokenized:", [tokenizer.convert_ids_to_tokens(idx) for idx in X_train[0][i].tolist()])
    # print("idx: ", X_train[0][i])
    # print("mask: ", X_train[1][i])
    # print("segment: ", X_train[2][i])
    
	# Get model from fit_bert_classif
    model = None

    seed(1)
    random.set_seed(1)
    np.random.seed(1)

    print("-----Train and Test-----")
    # this takes a while
	# unfortunately my computer could only do batch size 2 for bert, increase the batch size will certainly fasten the model training
    model, predicted_prob, predicted = fit_bert_classif(X_train, y_train, X_test, y_test, encode_y=True,
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
    resultFile = os.path.join(outputDir, 'results.txt')
    print("-----Start Evaluation-----")
    evaluate_multi_classif(y_test, predicted, predicted_prob, outputDir, resultFile)
