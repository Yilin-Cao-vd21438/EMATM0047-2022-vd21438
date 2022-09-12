import nltk
# May have to download first time
# nltk.download('averaged_perceptron_tagger')

## for decision trees
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from dtreeviz.trees import dtreeviz # remember to load the package
import sklearn 
#from sklearn.datasets import *


# https://stackoverflow.com/questions/39634222/is-there-a-way-to-remove-proper-nouns-from-a-sentence-using-python/39635503
# Example to show how to remove proper nouns from a text segment
# Not greatest approach, cannot handle "named entity" of several words.
# Now, just as a warning, POS tagging is not 100% accurate and may mistag some ambiguous words.
# Also, you will not capture Named Entities in this way as they are multiword in nature.
# "New York City is a cool city" -> "is a cool city"
# "New york city is a cool city" -> "york city is a cool city"

sentence = "It burns worse than his hand did when he was thirteen and still mildly protected. That burn still aches when it gets too cold or when he flexes his index finger too much. But this is a different type of burning. For just a second, he understands how Katniss felt in the arena as it burned down around her. And he can't breathe, but Peeta pushes through the eye-watering pain.  New york city is a cool city"
print( sentence )
tagged_sentence = nltk.tag.pos_tag(sentence.split())
#edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
edited_sentence = []
for word,tag in tagged_sentence:
    if tag != 'NNP' and tag != 'NNPS':
        edited_sentence.append(word)
print(' '.join(edited_sentence))


classifier = tree.DecisionTreeClassifier(max_depth=2)  # limit depth of tree
iris = sklearn.datasets.load_iris()
classifier.fit(iris.data, iris.target)


print( iris.target )

viz = dtreeviz(classifier, 
               iris.data, 
               iris.target,
               target_name='variety',
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"]  # need class_names for classifier
              )  
              
viz.view() 
