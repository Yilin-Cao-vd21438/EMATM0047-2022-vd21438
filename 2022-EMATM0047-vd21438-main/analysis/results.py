
import sys
import os
from instream import InStream

class Series:
    def __init__(self, name):
        self._name = name
        self._map = dict()

    def getName(self):
        return self._name

    def put(self, key, value):
        self._map[key] = value

    def get(self, key):
        return self._map[key]

    def hasKey(self, key):
        return key in self._map

    def keys(self):
        return sorted( self._map.keys() )

def process( serie, resultpath, numberofsentences ):
    #basenameext = os.path.basename(resultpath)
    #basename    = os.path.splitext(basenameext)[0]
    inputfile = InStream(resultpath)
    #print( '    ' + str( basename ) )
    while( inputfile.hasNextLine() ):
        line = inputfile.readLine()
        #Accuracy:0.89
        #Auc:0.96
        if( line.startswith('Accuracy') ):
            tokens = line.split(':')
            accuracy = float(tokens[1])
            serie.put(numberofsentences, accuracy)
            #print( '    ' + str( basename )  + ' ' + str(numberofsentences) + ' -> ' + str(accuracy) )
    del(inputfile)


#=======================================================================
# Entry point
#=======================================================================
def main():
    # series is a dict that maps name -> series
    # series maps int -> float
    series = dict() 

    resultsDir = sys.argv[1]
    for approach in os.listdir( resultsDir ):
        #print( approach )

        # Go ahead and make sure we have a seriesmap
        if( approach not in series ):
            serie = Series( approach )
            series[serie.getName()] = serie
        serie = series[approach]

        approachDir = os.path.join( resultsDir, approach )
        for scenario in os.listdir( approachDir ):
            # parse out the number of sentences, e.g. scenario='arff_6'
            tokens = scenario.split('_')
            numberofsentences = int(tokens[1])
            #print( '  ' + scenario + ' -> ' + str(numberofsentences) )
            
            scenarioDir = os.path.join( approachDir , scenario )
            for resultfile in os.listdir( scenarioDir ):
                resultpath = os.path.join( scenarioDir , resultfile )
                if( 'results.txt' == resultfile ):
                    process( serie, resultpath, numberofsentences )
                #print( '    ' + resultfile )
          
    for approach in series.keys():
        serie = series[approach]
        print( approach )
        for numberofsentences in serie.keys():
            accuracy = serie.get(numberofsentences)
            #print( '    ' + str( approach )  + ' ' + str(numberofsentences) + ' -> ' + str(accuracy) )
            print( '(' + str(numberofsentences) + ', ' + str(accuracy) + ')' )
      

if __name__ == '__main__':
    main()


