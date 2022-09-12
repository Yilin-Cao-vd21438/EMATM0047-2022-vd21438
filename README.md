# EMATM0047-2022-vd21438
This is the program used in the msc thesis of Yilin Cao-vd21438.

Webcrawler > WorksCrawler.java is the working main for data collection. <br/>
Analysis > nlp.py is the working main for data analysis. <br/>

The NLP Models evaluated are: <br/>
* Gensim Word2vec > nlp_word2vec.py
* Pretrained DistilBert > example_bert_correct.py
* Pretrained DistilBert with TFAutoModelForSequenceClassification > example_bert.py
* Pretrained Bert(full) > example_bert_full_old.py

TFAutoModelForSequenceClassification brought the the best classification result with a f1-score of 0.97 given 5-10 sentences per instance.
