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



class DataProcessor:
    # This is the class to load and preprocess input data

    def __init__(self,configfile):
        # This is the first method in the DataProcessor class
        self.config = configfile

    def preprocessor(self):
        # This is the method to first load all the files from the path

        Resume_File_Path = self.config.get('DataFiles','resumeData')
        # Starting two empty lists to load the pdf documents and word document lists
        print('This is the resume file path',Resume_File_Path)
        pdf_list = []
        doc_list = []
        for (path,_,files) in os.walk(Resume_File_Path):
            # In the above walk you get the full path of the resume like /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/Datasets/ResumeData and
            # The files in the path in the form of a list.
            for file in files:
                # Take one file at a time and check if they are pdf or docx
                ext = os.path.splitext(file)[-1]
                if ext == '.docx':
                    doc_list.append(path+'/'+file)
                elif ext == '.pdf':
                    pdf_list.append(path+'/'+file)

        return pdf_list,doc_list

    def pdfProcessor(self):
        # Get the pdf files from the method preprocessor
        pdf_list,_ = self.preprocessor()
        # Loop over all the pdf files
        pdfResume = []
        for pdfFile in pdf_list:
            # Convert the pdf file into multiple images
            pages = convert_from_path(pdfFile, 500)
            # Starting an empty list to store the text details
            resDetails = []
            for page in pages:
                 # Recognize the text as string in image using pytesserct
                 text = str(((pytesseract.image_to_string(page))))
                 # Splitting the text based on sentences
                 text = text.split('.')
                 resDetails.append(text)
            # Appending the overall details of all consumers into the bigger list
            pdfResume.append(resDetails)
        return pdfResume




