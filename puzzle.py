from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np 
import cv2
import imutils

def find_puzzle(image,debug=False):
    # convert the image to the grayscale and blur it slightly
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),3)

    # apply adaptive thresholding

    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    thresh = cv2.bitwise_not(thresh)

    #  check to see if we are visualizing each step of the image


    if debug:
        cv2.imshow("Puzzle Thresh" ,thresh)
        cv2.waitKey(0)
    
    ## finf contours in threshold image

    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts= imutils.grab_contours(cnts)

    

    # initialize a contour thet corresponds to the puzzle outline
    puzzleCnt = None

    for c in cnts : 
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)

        # if our aprox. contour has four points then we can 
        # assume we have the outline of the puzzle

        if len(approx)==4:
            puzzleCnt = approx
            break

    
    # if the puzzle contour is empty then our script could not find 
    # the outline of the Sudoku so raise an Error.

    if puzzleCnt is None:
        return (None,None)


    if debug : 

        output = image.copy()
        cv2.drawContours(output,[puzzleCnt],-1,(0,255,0),2)
        cv2.imshow("Puzzle outline",output)
        cv2.waitKey(0)
    

    # apply a four point perspective transform to both original and
    # grayscale image to obtain a top-down bird's eye view
    # of puzzle
    puzzle = four_point_transform(image,puzzleCnt.reshape(4,2))
    warped = four_point_transform(gray,puzzleCnt.reshape(4,2))

    # check to see if we are visualizing the perspective

    if debug:
        cv2.imshow("Puzzle Transform",warped)
        cv2.waitKey(0)

    return (puzzle,warped) 




def extract_digits(cell,debug=False):
    #apply automatic thresholdig to the cell and then clear any 
    #connected borders that touch the border of the cell
    if cell is None:
        return None
    thresh = cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]
    thresh = clear_border(thresh)

    if debug:
        cv2.imshow("Cell thresh",thresh)
        cv2.waitKey(0)

    # find contours int he threshold cell
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    #if no contours then empty cell
    if len(cnts)==0:
        return None

    # otherwise find the largest contour in the cell and create a mask for 
    #this contour

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape,dtype="uint8")
    cv2.drawContours(mask,[c],-1,255,-1)

    # compute percentage of masked pixels relative to the total
    # area of the image

    (h,w) = thresh.shape
    percentageFilled = cv2.countNonZero(mask)/float(w*h)

    # if less than 3% of mask then we are looking at noise
    #and safely ignore the contour

    if percentageFilled<0.03:
        return None
    
    digit = cv2.bitwise_and(thresh,thresh,mask=mask)

    # check to see if we should visualizze the masking

    if debug:
        cv2.imshow("Digit",digit)
        cv2.waitKey(0)
    return digit



