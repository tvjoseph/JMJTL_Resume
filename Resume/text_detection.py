'''
JMJPFU
27-Feb-2020
THis is the script for text detection algorithm

Lord bless this attempt of yours
'''

# IMporting the required packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',type=str,help = 'path to the input image')
ap.add_argument('-east','--east',type=str,help='Path to input east text detector')
ap.add_argument('-c','--min-confidence',type=float,default=0.5,help='minimum probability required to inspect a region')
ap.add_argument('-w','--width',type=int,default=320,help='Resized image width. THis should be a multiple of 32')
ap.add_argument('-e','--height',type=int,default=320,help='resized image height')
args = vars(ap.parse_args())

# Loading the image and resizing them

image = cv2.imread(args['image'])
orig = image.copy()
(H,W) = image.shape[:2]

# Getting the new height and width and determine the ratio in change for them

(newW,newH) = (args['width'],args['height'])

rW = W / float(newW)
rH = H / float(newH)

# Resizing the image and grab the new dimensions

image = cv2.resize(image,[newW,newH])
(H,W) = image.shape[:2]

# Getting two output layers from the East text detector

layerNames = ['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']

# Opening the Text detector
print('[INFO] loading East text detector .....')
net = cv2.dnn.readNet(args['east'])
# Capturing a blob from the image and performing a forward pass on the network

blob = cv2.dnn.blobFromImage(image,1.0,(W,H),(123.68,116.78,103.94),swapRB=True,crop=False)
start = time.time()
# Implementing a forward pass on the network
net.setInput(blob)
(scores,geometry) = net.forward(layerNames)
end = time.time()

print('[INFO] text detection took {:.6f} seconds'.format(end-start))
