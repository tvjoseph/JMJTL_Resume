'''
JMJPFU
13th April 2020
This is the script for the inference process. During the inference process we take some query and the return the best recommended resumes
Lord bless this attempt of yours
'''

import argparse
import sys
from configparser import ConfigParser
from Modelfiles import Model
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pymongo import MongoClient
import nltk
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
print('[INFO] completed creating the query vectors')

# Process 2 : Pick each of the resume vectors and then find the vector similarities and scores

client = MongoClient(port=27017)
#db=client.Subjects

db = client.Subjects

collections = db.processedResume
for collection in collections.find():
    print(collection)


