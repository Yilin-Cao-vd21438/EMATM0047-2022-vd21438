import os
import sys
import json

# gensim not up to task of preprocessing
import nltk.data
from nltk import sent_tokenize


def text2json( documentDict, top_directory, category, sentencesPerRecord ):
    """
    Reads all .txt files in the specified directory and convert into records.
    The records are added to the specified dictionary with a unique key 
    derived from the filename.  Each record is also annotated with the category.
    :parameter
        :param documentDict: dict - key to record mapping
        :param top_directory: str - name of directory containing text files
        :param category: str - category to annotate record with
        :param sentencesPerRecord: int - number of sentences to store in each record
    :return
        None
    """
    # Go through each text file and process
    #for root, dirs, files in os.walk(top_directory):
    #    for file in files:
    for book in os.listdir(top_directory):
        bookdir = os.path.join(top_directory, book)
        for file in os.listdir(bookdir):
            if( file.endswith('.txt') and file.startswith("pos") is False ):
                filename = os.path.basename(file)
                document = open(os.path.join(bookdir, file)).read() # read the entire document, as one big string
                print( "Document " + filename + " = "  + str(len(document)) )
 
                # Explicitly uses english sentence detector punkt
                #sentences = sent_detector.tokenize( document )
                # Implicit, frankly not sure how this works, perhaps language autodetect then use punkt?
                sentences = sent_tokenize( document )

                paragraph = list()
                paragraphNumber = 0
                # Every 5 sentences create a record
                for sentence in sentences:
                    #sentence = sentence.rstrip('\n') # remove newlines
                    sentence = sentence.replace('\n',' ')
                    #sentence = sentence.encode('utf-8','ignore').decode("utf-8")
                    # DEBATABLE: UTF allows many non-english symbols but may be informative.  This gets rid of them.
                    sentence = sentence.encode('ASCII','ignore').decode("ASCII")
                    paragraph.append( sentence )
                    if( len(paragraph) >= sentencesPerRecord ):
                        p = " ".join(paragraph)
                        #print( "    paragraph = "  + str(len(paragraph)) + " " + p)
                        key = filename + "_" + str(paragraphNumber)
                        record = dict()
                        record["category"] = category
                        record["headline"] = p
                        record["filename"] = filename
                        record["paragraph"] = paragraphNumber
                        documentDict[key] = record
                        paragraphNumber = paragraphNumber + 1
                        paragraph.clear()
                # Do not forget the last paragraph, may omit so we only have 5 sentence paragraphs?
                # We currently ignore the last (likely partial) paragraph.



def main():
    """
    Main entry point for program.  Takes any parameters from command line.
    """
    if( len(sys.argv) < 2 ):
        print( '  Usage: python text2json <sentences per record>' )
        print( 'Example: python text2json 5' )
        return

    maxSentencesPerRecord = int( sys.argv[1] )

    # For each number of sentences per record, process and produce file
    sentencesPerRecord  = 1
    step = 1
    while( sentencesPerRecord <= maxSentencesPerRecord ):
        documentDict = dict()
        text2json( documentDict, '.data/1_PrintFiction/', 'print', sentencesPerRecord )
        text2json( documentDict, '.data/2_DigitalFiction/', 'digital', sentencesPerRecord )
        
        #=======================================================================
        # Produce JSON file
        #=======================================================================
        filename = "arff_" + str(sentencesPerRecord) + ".json"
        print( "Writing to JSON file " + filename )
        file_handle = open( filename, "w+")
        # Sadly cannot use default format for dictionary to json
        # The gensim loader expects records with no commas
        #json.dump(documentDict, file_handle, indent=4)
        for key in documentDict.keys():
            record = documentDict[key]
            # record is a dictionary and should be correct format
            #recordStr = str(record) # assume this is json? single quotes, need double quotes
            recordStr = json.dumps( record  )
            # one record per line enclosed in braces, no comma
            # Sadly (yet again), gensim does not like single quotes for strings, convert to double.
            # We need to be careful not to


            file_handle.write( recordStr )
            file_handle.write( "\n" ) 
        file_handle.close()
        
        # increment for the next file
        sentencesPerRecord = sentencesPerRecord + step

#=======================================================================
# Command line bootstrap
#=======================================================================
if __name__ == '__main__':
    main()
