# EMATM0047-vd21438
This is the program used in Data Science MSc thesis of Yilin Cao-vd21438.

Webcrawler > WorksCrawler.java is the working main for data collection. Data will be scraped from fan fiction website archiveofourown. <br/>
Analysis > nlp.py is the working main for data analysis. <br/>

The NLP Models evaluated are: <br/>
* Gensim Word2vec > nlp_word2vec.py
* Pretrained DistilBert > example_bert_correct.py
* Pretrained DistilBert with TFAutoModelForSequenceClassification > example_bert.py
* Pretrained Bert(full) > example_bert_full_old.py

TFAutoModelForSequenceClassification brought the the best classification result with a f1-score of 0.96 given 5-20 sentences per instance.
