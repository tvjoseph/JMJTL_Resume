'''
JMJPFU
13th April 2020
This is the script for the inference process. During the inference process we take some query and the return the best recommended resumes
Lord bless this attempt of yours
'''

import argparse
import sys
from configparser import ConfigParser
from Modelfiles import Model,scoring,resumeReco
from Datasets import DataProcessor
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pymongo import MongoClient
import nltk
import numpy as np
import pandas as pd

#nltk.download('punkt')


# Parsing the arguments
# Adding the arguments for the project
ap = argparse.ArgumentParser()
ap.add_argument('-C','--configfile',required=True,help='This is the path to the configuration files')
args = vars(ap.parse_args())

# Getting the configuration in place

print('System arguments path',sys.argv)

sys_arguments = sys.argv
default_cfg = ConfigParser()
default_cfg.read(sys_arguments[2])

# First import the bert model and evaluate it

print('[INFO] loading and initializing BERT models......')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Process 1 : Let us load the Query from the configuration file and convert them into vectors

# Instantiating the model class

modelBuilding = Model(model,tokenizer)


resQuery = default_cfg.get('ResumeQuery','resQuery')

# Tokenizing the queries
print('[INFO] Creating the vectors of the queries.......')
query_list = nltk.tokenize.sent_tokenize(resQuery)
qeryDf = modelBuilding.qvectorMaker(query_list)
print(qeryDf.shape)
print('[INFO] completed creating the query vectors')

# Process 2 : Pick each of the resume vectors and then find the vector similarities and scores

# Defining the mongo db credentials
client = MongoClient(port=27017)
db = client.Subjects
collections = db.processedResumeVec
# Instantiate the data loader class
dl = DataProcessor(default_cfg)

# Start the process of processing each of the vectors in the collection vector
# Create an empty dictionary to store the scores

print('[INFO] Starting the resume scoring process')

scoredRes = scoring(dl,collections,qeryDf)

# Next start the process for recommending the resumes

print('[INFO] Starting the resume recommendation process')

recomendations = resumeReco(scoredRes,default_cfg)

print(recomendations)






'''

# Storing the scores in the pickle file

print('[INFO] Starting to save the resDF as pickle file')

output_File_Path = default_cfg.get('Outputfiles','scrapDocs')
resFilepath = output_File_Path + '/' + "resDF.pickle"
resDf_vector = open(resFilepath,"wb")
pickle.dump(resScores, resDf_vector)
resDf_vector.close()

'''






