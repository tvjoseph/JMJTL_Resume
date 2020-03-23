'''
JMJPFU
27-Feb-2020
This is the script for the resume OCR part of the Resume recommendation project
Lord bless this attempt of yours
'''

# Importing the required libraries

from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2

# Implementing the text detection in a function
def decode_predictions(scores,geometry):
    # Grab the number of rows and columns from the network
    (numRows,numCols) = scores.shape[2:4]
    # Defining the placeholders for rectangles and its confidence values
    rects = []
    confidences = []

    # Looping over the numver of rows
    for y in range(0,numRows):
        # Extracting score probabilities and the x, y coordinates for extracging bounding boxes
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]

        # Looping over the number of columns
        for x in range(0,numCols):
            # If the score dosent have enough probability, ignore it

            if scoresData[x] < args['min_confidence']:
                continue
            # Compute the offset factor as the resulting map will be 4x smaller than the input
            (offsetX,offsetY) = (x * 4.0,y*4)

            # Extracging the rotation angle for the prediction and compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Use the geometry volume to extract the width and height of the image
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute both the starting and ending (x,y) coordinates for the text bounding boxes

            endX = int(offsetX + (cos * xData1[x])  +  (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add bounding box co-ordinates and probability scores to the respective lists created
            rects.append((startX,startY,endX,endY))
            confidences.append(scoresData[x])

    return (rects,confidences)

# Constructing the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',type=str,help='path to the input image')
ap.add_argument('-east','--east',type=str,help= 'path to input East text detector')
ap.add_argument('-c','--min-confidence',type=float,default=0.5,help='Minimum probability required to inspect a region')
ap.add_argument('-w','--width',type=int,default=320,help='nearest multiple of 32 for resized width')
ap.add_argument('-e','--height',type=int,default=320,help='nearest multiple of 32 for resized image')
ap.add_argument('-p','--padding',type=float,default=0.0,help='amount of padding to be added to each border of ROI')
args = vars(ap.parse_args())

# Loading the input image and getting the dimensions

image = cv2.imread(args['image'])
orig = image.copy()
(origH,origW) = image.shape[:2]

# Setting the new width and height and then determine the ratio in change for both width and height

(newW,newH) = (args['width'],args['height'])
rW = origW / float(newW)
rH = origH/float(newH)

# Resize the image and grab the new image dimensions
image = cv2.resize(image,(newW,newH))
(H,W) = image.shape[:2]

### Using the East text detection algorithm to get the layers of the model relevant for the probabilities and the coordinates of the bounding boxes

layerNames = ['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']

# Load the pre-trained EAST text detector
print('[INFO] loading East text detector......')
net = cv2.dnn.readNet(args['east'])

# Get a blob from the image to pass through the network

blob = cv2.dnn.blobFromImage(image,1.0,(W,H),(123.68,116.78,103.94),swapRB=True,crop=False)

net.setInput(blob)

(scores,geometry) = net.forward(layerNames)

# Decoding the predictions and applying non-maxima suppression to suppress weak overlapping bounding boxes

(rects,confidences) = decode_predictions(scores,geometry)
boxes = non_max_suppression(np.array(rects),probs=confidences)

print('These are the boxes',boxes)

# Getting the list of results

results = []

# Looping over the bounding boxes

for (startX,startY,endX,endY) in boxes:
    # Scaling the bounding boxes based on the respective ratios

    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Compute the deltas around the bounding boxes to pad the bounding boxes

    dX = int((endX - startX) * args['padding'])
    dY = int((endY - startY) * args['padding'])

    # Applying padding to each side of the bounding boxes respectively

    startX = max(0,startX - dX)
    startY = max(0,startY - dY)

    endX = min(origW,endX + (dX * 2))
    endY = min(origH,endY + (dY * 2))

    # Extracting the actual padded ROI

    roi = orig[startY:endY,startX:endX]

    # Using tesseract to predict the text from the bounding boxes

    config = ('-l eng --oem 1 --psm 4')
    text = pytesseract.image_to_string(roi,config=config)

    # Adding the bounding boxes the co-ordinates and OCR'd text to the list

    results.append(((startX,startY,endX,endY),text))

# Sorting the result bouding boxes from top to bottom

results = sorted(results,key=lambda r:r[0][1])

# Looping over the results

for ((startX,startY,endX,endY),text ) in results:
    # Displaying the text OCR ed by Tesseract

    print('OCR Text')
    print('==============')
    print('{}\n'.format(text))

    # Stripping out the non-ASCII text so that we can draw the text on the image

    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()
    cv2.rectangle(output,(startX,startY),(endX,endY),(0,0,255),2)
    cv2.putText(output,text,(startX,startY - 20),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

    # Show the output image
    cv2.imshow('TExt detection',output)
    cv2.waitKey(0)

