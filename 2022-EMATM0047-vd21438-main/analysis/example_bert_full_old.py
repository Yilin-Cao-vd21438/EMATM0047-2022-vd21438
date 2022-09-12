from nlp_utils import *
import numpy as np
import transformers
from tensorflow.keras import models, layers, preprocessing
from tensorflow import random
from numpy.random import seed

seed(1)
random.set_seed(1)
np.random.seed(1)


def evaluate(dtf_train, dtf_test, y_train, y_test):
    seed(1)
    random.set_seed(1)
    np.random.seed(1)
    dtf_train = dtf_train.copy()
    dtf_test = dtf_test.copy()

    y_train = dtf_train["y"].values
    y_test = dtf_test["y"].values

    # Load pre-trained BERT tokenizer (full)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')
    # pre-trained bert with config
    config = transformers.BertConfig(dropout=0.2, attention_dropout=0.2, output_hidden_states=False,
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

    X_train = tokenize_bert(corpus=dtf_train["text"], tokenizer=tokenizer, maxlen=50)
    X_test = tokenize_bert(corpus=dtf_test["text"], tokenizer=tokenizer, maxlen=50)

    # check feature creation
    i = 0
    print("-----Check feature creation-----")
    print("txt: ", dtf_train["text"].iloc[0])
    print("tokenized:", [tokenizer.convert_ids_to_tokens(idx) for idx in X_train[0][i].tolist()])
    print("idx: ", X_train[0][i])
    print("mask: ", X_train[1][i])
    print("segment: ", X_train[2][i])

    # Pre-trained BERT + fine-tuning (transfer learning)

    ## inputs
    idx = layers.Input(50, dtype="int32", name="input_idx")
    masks = layers.Input(50, dtype="int32", name="input_masks")
    nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased', config=config)
    bert_out = nlp(idx, attention_mask=masks)[0]
    ## fine-tuning
    x = layers.GlobalAveragePooling1D()(bert_out)
    x = layers.Dense(64, activation="relu")(x)
    y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)
    ## compile
    model = models.Model([idx, masks], y_out)
    for layer in model.layers[:3]:
        layer.trainable = False
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    seed(1)
    random.set_seed(1)
    np.random.seed(1)

    print("-----Train and Test-----")
    # this takes a while
    model, predicted_prob, predicted = fit_bert_classif(X_train, y_train, X_test, encode_y=True,
                                                        model=model, epochs=1, batch_size=64)
														
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
	outputDir = "results/bert_cnn/arff_20/"
    resultFile = "results/bert_cnn/arff_20/results.txt"
    print("-----Start Evaluation-----")
    evaluate_multi_classif(y_test, predicted, predicted_prob, outputDir, resultFile)
