'''
JMJPFU
25-Mar-2020
This is the script for the resume driver project
Lord bless this attempt of yours
'''

import argparse
import sys
from configparser import ConfigParser

from Datasets import DataProcessor



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
dl = DataProcessor(default_cfg)

pdf_list = dl.pdfProcessor()
print('PDF list',len(pdf_list))
print(pdf_list[0])

#input_data = preprocess(default_cfg)