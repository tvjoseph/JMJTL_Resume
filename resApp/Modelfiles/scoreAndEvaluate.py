'''
JMJPFU
17-April-2020
THis is the modified script for scoring and evaluating the collections and also giving the recommendations.
The dicProcessor is brought to this script from the data loader script. Also all references to the
'''
import pandas as pd
import numpy as np
import operator
from sklearn.metrics.pairwise import cosine_similarity

# This is the method to process each of the collections from Mongodb and the create similarity vectors
def dicProcessor(collection):
    # Collection is a list of vectors which is passed for processing

    # Create an empty array same size as the collection

    resCon = np.empty((len(collection), 768), dtype='float')

    # First find the number of vectors in the collection and start a for loop
    for i in range(len(collection)):
        # First remove the two square braces on both sides
        numpStr = collection[i][2:-2]
        # Convert the string array into a proper array
        numpar = np.fromstring(numpStr, dtype=float, sep=' ')

        # Store the array in the empty array
        resCon[i,:] = numpar

    return resCon

def resSimCreate(resConDf,qeryDf):
    # Find the similarity score for these vectors
    simDf = pd.DataFrame(cosine_similarity(qeryDf, resConDf))
    simDf = simDf.T
    return simDf

resScores = {}

def scoring(collections,qeryDf):
    # dl is the data loader class
    # collections is the mongo db collection which contains the vectors
    for collection in collections.find():
        resName = collection['ResumeName'][0]
        # Create an empty data frame
        resDf = pd.DataFrame()
        # First take the 'Academic_credentials'
        if len(collection['Academic_credentials']) > 0:
            resConDf = dicProcessor(collection['Academic_credentials'])
            resDf = pd.concat([resDf,resSimCreate(resConDf,qeryDf)])
        # Take the 'Certifications'
        if len(collection['Certifications']) > 0:
            resConDf = dicProcessor(collection['Certifications'])
            resDf = pd.concat([resDf, resSimCreate(resConDf, qeryDf)])
        # Take the 'Key_projects'
        if len(collection['Key_projects']) > 0:
            resConDf = dicProcessor(collection['Key_projects'])
            resDf = pd.concat([resDf, resSimCreate(resConDf, qeryDf)])
        # Take the 'Profile_summary'
        if len(collection['Profile_summary']) > 0:
            resConDf = dicProcessor(collection['Profile_summary'])
            resDf = pd.concat([resDf, resSimCreate(resConDf, qeryDf)])
        # Take the 'Skills'
        if len(collection['Skills']) > 0:
            resConDf = dicProcessor(collection['Skills'])
            resDf = pd.concat([resDf, resSimCreate(resConDf, qeryDf)])
        # Take the 'Work_exp'
        if len(collection['Work_exp']) > 0:
            resConDf = dicProcessor(collection['Work_exp'])
            resDf = pd.concat([resDf, resSimCreate(resConDf, qeryDf)])
        # Take the 'accolades'
        if len(collection['accolades']) > 0:
            resConDf = dicProcessor(collection['accolades'])
            resDf = pd.concat([resDf, resSimCreate(resConDf, qeryDf)])
        resScores[resName] = resDf
    return resScores

def resumeReco(Scores,default_cfg):
    # Scores is the dictionary which contains the scores for all the resumes based on the number of queries

    # First consolidate all the values of the dictionary

    # To find the quantile values of the resume score
    resScores = pd.DataFrame()
    for value in Scores.values():
        resScores = pd.concat([resScores, pd.DataFrame(value)])
    # Get the required threshold value for calculting the quantiles
    resQuery = [float(default_cfg.get('ResumeQuery', 'thresh'))]
    # Get the quantile values for the resume scores
    qtvals = resScores.quantile(q=resQuery, axis=0).values
    # Define an empty dictionary to store the consolidated scores
    resRec = {}

    for key in Scores.keys():
        # Take the resume similarity scores from the dictionary
        tempDf = pd.DataFrame(Scores[key].values)
        # Take the count of those scores which are greater than the quantile values for each query and find its proportion
        threshCount = ((tempDf[tempDf > qtvals].count().values) / len(tempDf)) * 100
        # Take the sum of these concolidated scores by weighting each query
        # First get the weighting parameters from the config file
        wts = [float(wt) for wt in default_cfg.get('ResumeQuery', 'weights').split(',')]
        # Multiply the counts of thresholds with the weights and sum them up to get the consolidated score
        consolScore = sum(threshCount * wts)
        # Store the consolidated score in a dictionary
        resRec[key] = consolScore
        # Sort the resumes based on the scores
        resRecomends = sorted(resRec.items(), key=operator.itemgetter(1), reverse=True)
    return resRecomends