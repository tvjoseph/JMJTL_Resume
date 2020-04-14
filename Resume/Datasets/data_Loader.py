'''
JMJPFU
25-Mar-2020
This is the data loading script which seperates the different input documents and the reads the content

Lord bless this attempt of yours
'''
import os
from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path
import docx
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



class DataProcessor:
    # This is the class to load and preprocess input data

    def __init__(self,configfile):
        # This is the first method in the DataProcessor class
        self.config = configfile
        self.nlp = spacy.load('en_core_web_sm')
        #self.tokenizer = bertTokenizer
        #self.model = bertModel
    '''    
    The below method looks at the path where all the files reside and then return the path of all the files in a list 
    for the next process
    '''
    def preprocessor(self):
        # This is the method to first load all the files from the path

        Resume_File_Path = self.config.get('DataFiles','resumeData')
        # Starting two empty lists to load the pdf documents and word document lists
        print('This is the resume file path',Resume_File_Path)
        pdf_list = []
        pdf_names = []
        doc_list = []
        doc_names = []
        for (path,_,files) in os.walk(Resume_File_Path):
            # In the above walk you get the full path of the resume like /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/Datasets/ResumeData and
            # The files in the path in the form of a list.
            for file in files:
                # Take one file at a time and check if they are pdf or docx
                ext = os.path.splitext(file)[-1]
                if ext == '.docx':
                    doc_list.append(path+'/'+file)
                    doc_names.append(os.path.splitext(file)[-2])
                elif ext == '.pdf':
                    pdf_list.append(path+'/'+file)
                    pdf_names.append(os.path.splitext(file)[-2])

        return pdf_list,pdf_names,doc_list,doc_names

    '''    
        The below method is a utility function which is used in the subsequent program. This method splits any strings as per the new lines
        and then joins them together
    '''

    def textProcessor(self,text):
        # Splitting the text based on sentences
        text = text.split('\n\n')
        textList = []
        for strings in text:
            tempText = strings.split('\n')
            tempText = ' '.join(tempText)
            textList.append(tempText)
        return textList

    '''    
        The below method implements the Named entity recognition method for text in the resume data list
    '''

    # The below is the function to create the NER from the text
    def nerMaker(self,indResume):
        # indResume is the list of resume text generated from pdf documents and docx documents
        # Define a dictionary to store the values
        resNer = {}
        # Loop through each of the text in the indResume list
        for resline in indResume:
            # Parse through the self.nlp spacy object defined in the init program
            doc = self.nlp(resline)
            # Extract the labels and its entity
            for X in doc.ents:
                if X.label_ in resNer:
                    resNer[X.label_].append(X.text)
                else:
                    resNer[X.label_] = [X.text]
        return resNer

    '''    
     The below method is an important method for processing pdf documents.
    '''

    def pdfProcessor(self):
        # Get the pdf files from the method preprocessor
        pdf_list,pdf_names,_,_ = self.preprocessor()
        # Loop over all the pdf files
        pdfResume = []
        for i,pdfFile in enumerate(pdf_list):
            # Convert the pdf file into multiple images
            pages = convert_from_path(pdfFile, 500)
            # Starting an empty list to store the text details
            resDetails = []
            for page in pages:
                 # Recognize the text as string in image using pytesserct
                 text = str(((pytesseract.image_to_string(page))))
                 tempText = self.textProcessor(text)
                 for textline in tempText:
                     resDetails.append(textline)
            # Get the named entities from resDetails
            resNer = self.nerMaker(resDetails)
            resNer['ResumeName'] = pdf_names[i]
            resDetails.append(resNer)
            # Appending the overall details of all consumers into the bigger list
            pdfResume.append(resDetails)
        return pdfResume

    def docxProcessor(self):
        # Get the pdf files from the method preprocessor
        _,_,doc_list,doc_names = self.preprocessor()
        # Loop over all the doc files
        docResume = []
        for i,docFile in enumerate(doc_list):
            # Read the document in multiple docs
            doc = docx.Document(docFile)
            # Starting an empty list to store the text details
            indResume = []
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
                '\n'.join(fullText)
            # Removing empty spaces
            list2 = [e for e in fullText if e]
            # Further processing for tab spaced docs
            for list in list2:
                list = list.split('\t')
                for sublist in list:
                    list3 = "".join(sublist)
                    indResume.append(list3)
            # Removing blank spaces in the resume list
            indResume = [e for e in indResume if e]
            # Get the named entities from resDetails
            resNer = self.nerMaker(indResume)
            resNer['ResumeName'] = doc_names[i]
            indResume.append(resNer)
            # Appending the individual resumes
            docResume.append(indResume)

        return docResume

    # This method to be completed

    def allDocs(self):
        # This is a method to consolidate all the records
        pdfDocs = self.pdfProcessor()
        docDocs = self.docxProcessor()

        return pdfDocs,docDocs
    # This is the method to process each of the collections from Mongodb and the create similarity vectors
    def dicProcessor(self,collection):
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

    def resSimCreate(self,resConDf,qeryDf):
        # Find the similarity score for these vectors
        simDf = pd.DataFrame(cosine_similarity(qeryDf, resConDf))
        simDf = simDf.T
        return simDf






