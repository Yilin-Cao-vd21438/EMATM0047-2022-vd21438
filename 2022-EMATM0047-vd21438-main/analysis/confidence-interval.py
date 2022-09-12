import numpy as np
import scipy.stats as st

from instream import InStream

import os
import sys
import math

def ci_mean(data, alpha=0.95):
    # https://www.kite.com/python/answers/how-to-compute-the-confidence-interval-of-a-sample-statistic-in-python
    # https://www.kite.com/python/examples/702/scipy-compute-a-confidence-interval-from-a-dataset

    #define sample data
    #data = [12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29]

    #data = [0.72,0.74, 0.73, 0.70, 0.75]

    #create 95% confidence interval for population mean weight from t-distribution
    ci = st.t.interval(alpha=alpha, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    #print( 'CI=' + str(ci) )
    
    # So CI gives bound around mean, we want the MOE
    # CI=(0.7041161161190002, 0.7518838838809998)
    u = np.mean(data)
    
    
    #hiDiff = ci[1] - u
    #lowDiff = u - ci[0]
    #print( 'hiDiff ' + str(hiDiff) )
    #print( 'lowDiff ' + str(lowDiff) )

    moe = ci[1] - u # can use just upperbound, same diff from u to lower bound
    return moe
    

def ci_proportion(p_hat_list, confidenceLevel):
    confidenceLevels = { 90:1.64, 95:1.96, 98:2.33, 99:2.58 }

    p_hat = np.mean( p_hat_list )

    n = len( p_hat_list )

    SE_hat_p = np.sqrt(p_hat*(1-p_hat)/n)
    moe = 2 * SE_hat_p
    lb = np.round( p_hat - moe, 2 )
    ub = np.round( p_hat + moe, 2 )

    print('With 95% confidence between {} and {}'.format(lb, ub) )



def parseAccuracy( resultFile ):
    stream = InStream(resultFile)
    accuracy = None
    while( not stream.isEmpty() ):
        line = stream.readLine()
        #line = line.lstrip() # funny character in front of tdidf result.txt
        line = line.strip()
        if( line.startswith('Accuracy:') ):
            #print( '        ' + line )
            # Accuracy:0.68
            # Accuracy:0.89
            tokens = line.split(':')
            accuracy = float(tokens[1])
            break
    stream.close()
    return accuracy


def parseProcessingTime( resultFile ):
    stream = InStream(resultFile)
    processingTime = None
    while( not stream.isEmpty() ):
        line = stream.readLine()
        #line = line.lstrip() # funny character in front of tdidf result.txt
        line = line.strip()
        if( line.startswith('classification_processing_time=') ):
            #print( '        ' + line )
            # classification_processing_time=1434.2269115447998
            # Accuracy:0.89
            tokens = line.split('=')
            processingTime = float(tokens[1])
            break
    stream.close()
    return processingTime


def main():
    #data = [0.72,0.74, 0.73, 0.70, 0.75]
    ###ci_proportion( data, 95 )
    #ci_mean(data)

    if( len(sys.argv) != 2 ):
        print('Usage: <time | accuracy>')
        return
    computeTime = (sys.argv[1] == 'time')

    # Map series -> sentencesPerInstance -> list[x1, x2, x3, ..., xn]
    resultsMap = dict()
    

    verbose = False
    # List all files in a directory using os.listdir
    resultsDir = './results'
    for seriesName in os.listdir(resultsDir):
        seriesDir = os.path.join(resultsDir, seriesName)
        if not os.path.isfile(seriesDir):
            if(verbose): print(seriesDir)
            # Parse out the series, omitting the trial
            # word2vec_3
            tokens = seriesName.split('_')
            series = tokens[0]
            if(verbose): print( series )
            if( series not in resultsMap ):
                print( 'ADDING ' + series )
                resultsMap[series] = dict()
            seriesMap = resultsMap[series]

            # This is a dir, find the results.txt file in the dir
            for arrfName in os.listdir(seriesDir):
                arrfDir = os.path.join(seriesDir, arrfName)
                if(verbose): print( '    ' + arrfDir)
                # Parse out the number of sentences per instance in name of arrf dir
                # arff_4
                tokens = arrfName.split('_')
                sentencesPerInstance = int(tokens[1])
                if(verbose): print( '    ' + str(sentencesPerInstance) )

                if( sentencesPerInstance not in seriesMap ):
                    seriesMap[sentencesPerInstance] = list()
                sentencesSamples = seriesMap[sentencesPerInstance]

                if arrfName.startswith('arff_'):
                    for fileName in os.listdir(arrfDir):
                        resultFile = os.path.join(arrfDir, fileName)
                        if fileName.startswith('results.') and os.path.isfile(resultFile):
                            if(verbose): print( '        ' + resultFile)

                            metric = None
                            if( computeTime ):
                                metric = parseProcessingTime(resultFile)
                            else:
                                metric = parseAccuracy(resultFile)
                            
                            if(verbose): print( '        ' + str(metric) )
                            if metric is not None:
                                sentencesSamples.append(metric)
    print('===========================================')                    
    for series in resultsMap:
        seriesMap = resultsMap[series]
        print( series )
        for sentences in sorted(seriesMap.keys()):
            sentencesSamples = seriesMap[sentences]
            #print( '    ' + str(sentencesSamples) )
            if( len(sentencesSamples) > 0 ):
                u = round( np.mean(sentencesSamples), 4)
                moe = ci_mean( sentencesSamples )
                if(not moe > 0.0): moe = 0.0 # handles nan values
                moe = round(moe, 4)
                print( '({},{}) +-({},{})'.format(sentences, u, moe,moe) )



if __name__ == '__main__':
    main()
    

