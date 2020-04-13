'''
JMJPFU
25-Mar-2020
This is the script for the resume driver project
Lord bless this attempt of yours
'''

import argparse
import sys
from configparser import ConfigParser
from Modelfiles import Model
from Datasets import DataProcessor

import pickle
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pymongo import MongoClient

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging



# Adding the arguments for the project
ap = argparse.ArgumentParser()
ap.add_argument('-C','--configfile',required=True,help='This is the path to the configuration files')
args = vars(ap.parse_args())

# Getting the configuration in place

print('System arguments path',sys.argv)

sys_arguments = sys.argv
default_cfg = ConfigParser()
default_cfg.read(sys_arguments[2])



# Start the data loading module
'''
The data loading module will have the following structure
1. First load the folder where the files are kept and capture the file paths
2. It will first look at the extension of the files
3. If the file is a pdf then it will be processed in a certain way if a doc file another way.
4. Process the files and then store in a single data set with each line classified as to what category of feature it is

'''
# First import the bert model and evaluate it

print('[INFO] loading and initializing BERT models......')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

print('[INFO] loading the estimator models......')

# Importing the estimator
estimator_path = default_cfg.get('Modelfiles','estimator')
estimator_doc = open(estimator_path,"rb")
# Loading the estimator
estimator = pickle.load(estimator_doc)

print('[INFO] starting the data processing process .............')

dl = DataProcessor(default_cfg)

pdfDocs,docDocs = dl.allDocs()

# Instantiating the models

modelBuilding = Model(model,tokenizer)

#tempVec = modelBuilding.genMeanvec(str(docDocs[0][0]))

print('[INFO] Starting the consolidation of the resume documents')

# Consolidating all documents into one list
docTypes = [pdfDocs,docDocs]

# Opening the mongoDB client

client = MongoClient(port=27017)
db=client.Subjects

# Starting a for loop to go through each type of document

for doctype in docTypes:
    # Taking one of the resume format and generating two empty lists
    # The first list is to store the raw features
    # The second list is to store the vectors of the features
    allResume = []
    allResume_vector = []
    for resume in doctype:
        # Creating a dictionary for storing each customers details
        indResDic = {'ResumeName': [], 'Name': [], 'Academic_credentials': [], 'Certifications': [], 'Date': [],
                     'Key_projects': [], 'Profile_summary': [], 'Skills': [], 'Work_exp': [], 'accolades': [],
                     'email': [], 'location': [], 'phone': []}
        indResDic_docs = {'ResumeName': [], 'Name': [], 'Academic_credentials': [], 'Certifications': [],
                          'Date': [], 'Key_projects': [], 'Profile_summary': [], 'Skills': [], 'Work_exp': [],
                          'accolades': [], 'email': [], 'location': [], 'phone': []}
        # First get the resume Name
        indResDic['ResumeName'].append(resume[-1]['ResumeName'])
        indResDic_docs['ResumeName'].append(resume[-1]['ResumeName'])
        # Loop through all the text in the resume
        for i in range(0, len(resume[0:-1])):
            docs = resume[0:-1][i].split(';')
            for doc in docs:
                tempVector = modelBuilding.genMeanvec(str(doc))
                tempVector = tempVector.reshape((1, tempVector.shape[0]))
                pred = estimator.predict(tempVector)
                if pred[0] == 'BS':
                    continue
                else:
                    indResDic[pred[0]].append(tempVector)
                    indResDic_docs[pred[0]].append(doc)
        # Updating the last dictionary of NER components in the just created dictionary
        indResDic_docs.update(resume[-1])
        # inserting the record into the MOngo DB collections
        db.processedResume.insert_one(indResDic_docs)
        #db.processedResumeVec.insert_one(indResDic)
        # Inserting the details into the lists
    allResume.append(indResDic_docs)
    allResume_vector.append(indResDic)

print('[INFO] completed the processes. Printing results')

print('Resume lenght',len(allResume))
print(allResume[0])
#print('Doc list',len(docDocs))
#print(docDocs[0])

# Pickling The resume docs

##################################################################################

output_File_Path = default_cfg.get('Outputfiles','scrapDocs')

vecFilepath = output_File_Path + '/' + "vector.pickle"
docFilepath = output_File_Path + '/' + "rawDoc.pickle"

pickle_vector = open(vecFilepath,"wb")
pickle.dump(allResume_vector, pickle_vector)
pickle_vector.close()

pickle_doc = open(docFilepath,"wb")
pickle.dump(allResume, pickle_doc)
pickle_doc.close()

#input_data = preprocess(default_cfg)

