import sys

## for data
import json
import pandas as pd


def main():
    '''
    Utility to simply read a Pandas compatible JSON file and print number of records.
    '''
    if( len(sys.argv) < 2 ):
        print( '  Usage: python nlp <JSON filename>' )
        print( 'Example: python nlp arff_5.json' )
        return

    jsondataFilename = sys.argv[1]

    lst_dics = []
    with open(jsondataFilename, mode='r', errors='ignore') as json_file:
        for dic in json_file:
            lst_dics.append( json.loads(dic) )## print the first one


    ## The original dataset contains over 30 categories, but for the purposes of this tutorial,
    ## I will work with a subset of 3: Entertainment, Politics, and Tech.

    ## create dtf
    dtf = pd.DataFrame(lst_dics)## filter categories

    #--------------------------------------------------------------------#
    # Undersample to avoid class imbalance
    #--------------------------------------------------------------------#
    dtf.info(verbose=True)
    #print( dtf.describe(include='all') )
    groupedDataframe = dtf.groupby('category', group_keys=False).count()
    print( groupedDataframe )

    # iterating dataframe, row index and row values in dictionary 
    minCount = None
    for category, row in groupedDataframe.iterrows():
        categoryCount = row['headline']
        if( minCount is None or categoryCount < minCount ):
            minCount = categoryCount
    sampleSize = int(minCount * 0.80) # leave some randomness even in the minority class
    print( "sampleSize=" + str(sampleSize) )

#=======================================================================
# Command line bootstrap
#=======================================================================
if __name__ == '__main__':
    main()


