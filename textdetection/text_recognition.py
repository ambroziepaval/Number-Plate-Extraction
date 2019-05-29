from imutils.object_detection import non_max_suppression
from utils import text_filter

import numpy as np
import pytesseract
import argparse
import cv2
import re
import os


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding0
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have a minimum of 0.5 probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


def apply_pytesseract_predictions(image, rW, rH, boxes):
    """
    Extract the texts using pytesseract from the text boxes detected with the text detector.
    :param image:
    :param rH:
    :param rW:
    :param boxes: decoded predictions from the EAST text detector.
    :return: the list of all detected texts.
    """
    texts = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        (origH, origW) = image.shape[:2]
        # scale the bounding box coordinates based on the respective ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # remove extreme numbers
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(origW, endX)
        endY = min(origH, endY)

        # extract the actual padded ROI
        roi = image[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = "-l eng --oem 1 --psm 7"
        text = pytesseract.image_to_string(roi, config=config)

        texts.append(text)

    return texts


def extract_text(input_img):
    (origH, origW) = input_img.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    # use default of 320 & 320
    (newW, newH) = (320, 320)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(input_img, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that we are interested
    # the first is the output probabilities and the second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid",
                  "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] Loading pre-trained EAST text detector...")
    net = cv2.dnn.readNet(os.path.dirname(__file__) + "/frozen_east_text_detection.pb")

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # extract the text using pytesseract
    texts = apply_pytesseract_predictions(input_img.copy(), rW, rH, boxes)

    nprTextsFilter = text_filter.NprTextsFilter()
    dates, numbers = nprTextsFilter.filterDatesAndPlates(texts)
    return dates, numbers


image = cv2.imread('test_frame.png')
dates, numbers = extract_text(image)
print(dates)
print(numbers)
