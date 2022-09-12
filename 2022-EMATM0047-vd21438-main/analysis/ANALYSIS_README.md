README for the Computational lingustics pipeline.

There are two main sources of data extraction that can be accomplished with this programming project.

Print Data Type
The first data type is print fiction. This is data that has been scanned into the computer and saved as a PDF document. The scanned print fictions can be converted via OCR and have the text stripped from the image. OcrMain will save the resulting PNG and TXT file into the same location as the original PDF. An example of the command line function to OCR a PDF is:

java -cp ocrutil.jar OcrMain ../analysis/data/1_PublishedFictions/LionWitchWardrobe_CSLewis/


Digital Data Type
The second data type is digital fiction. This code specifically will crawl the website archiveofourown.org and visit the pages of the collections that have been saved into the collections.txt file located in ../webcrawler/


First step of digital data collection is to downloadDownload HTML
java -cp webcrawler.jar WorksCrawler 1 2 ../analysis/data/3_Fanfictions/
HTML to Text
java -cp ocrutil.jar Html2Text ../analysis/data/3_Fanfictions/Alex_Rider_-_Anthony_Horowitz/

#Create an arff from analysis directory
#java -cp weka.jar weka.core.converters.TextDirectoryLoader -dir data/foranalysis/ > analysis.arff


Bag of Word/Word Frequency Analysis
*Duplication of data to retain original structure/ for clarity of location, but not useful for the analysis with weka. Also keeping data integrity; By author type

Run weka with extra memory load
java -Xmx8192m -jar weka.jar
Rainbow stopwords


POS Analysis 
Text to POS - not needed to be done before making arff file/running through Weka // usefulness in its transparency
java -cp ocrutil.jar Text2Pos ../analysis/data/3_Fanfictions/Alex_Rider_-_Anthony_Horowitz/
POS tagging model from Stanford english-left3words-distsim.tagger; Used the Stanford POS library
 * at https://nlp.stanford.edu/software/tagger.shtml#Questions
 Rainbow stopwords
 Copy to posanalysis folder
 java -cp ocrutil.jar Copy

N-Gram Analysis
*Duplication of data to retain original structure/ for clarity of location, but not useful for the analysis with weka. Also keeping data integrity; data structure for analysis would likely be different/not by author type?
Create an arff from analysis directory
Don't want to get rid of stop words


**Weka - using to clean up - pull out words we don't want; probably want a file ultimately that would be of excluded words/stop words as well

java -cp ocrutil.jar Copy {pos|txt} ../analysis/data/1_PublishedFictions/ ../analysis/data/frequencyanalysis/


################################################################################
# POS TAG MEANINGS
################################################################################
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when


