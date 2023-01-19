import cv2
from edge_impulse_linux.image import ImageImpulseRunner
import numpy as np
import imutils


### Edge Detection ###
def edge_detect(img, view_edge):
    img_e = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_e = cv2.GaussianBlur(img_e, (7,7), 0) 
    img_e = cv2.Canny(img_e, 80, 200)
    indices = np.where(img_e == [255])
    img_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB)
    img = img_e if view_edge else img
    return img, indices



### Corner Detection ###
def corner_detect(img, view_edge):
    img_e, indices = edge_detect(img, True)
    img_e = np.float32(img_e)
    dst = cv2.cornerHarris(img_e,30,13,0.04)
    dst = cv2.dilate(dst,None)    # result is dilated for marking the corners, not important
    img_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB)
#    img[dst>0.1*dst.max()]=[0,0,255]

    # Blob Centroid Detection
    img_corners = np.zeros(img_e.shape, np.float32)
    img_corners = cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB)
    img_corners[dst>0.1*dst.max()]=[0,0,255]
    img_corners = np.uint8(img_corners)
    img_corners = cv2.cvtColor(img_corners, cv2.COLOR_BGR2GRAY)
    img_corners = cv2.GaussianBlur(img_corners, (5, 5), 0)
    img_corners = cv2.threshold(img_corners, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image
    cnts = cv2.findContours(img_corners.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    res_w = img_e.shape[0]
    res_h = img_e.shape[1]
    bl = (res_w, 0)
    tl = (res_w, res_h)
    br = (0, 0)
    tr = (0, res_h)
    # loop over the contours, keep bottom left and top left coordinates
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        # Find bottom left and top left corners
        if (cX + (res_h-cY)) < (bl[0] + (res_h-bl[1])):
            bl = (cX, cY)
        if (cX + cY) < (tl[0] + tl[1]):
            tl = (cX, cY)
        if (cX + cY) > (br[0] + br[1]):
            br = (cX, cY)
        if ((res_w-cX) + cY) < (res_w-(tr[0]) + tr[1]):
            tr = (cX, cY)

    img = img_e if view_edge else img
    return img, tl, tr, bl, br



### Classify Image ###
def img_classification(img, runner):
    # Convert to RGB and encapsulate raw values into array for model input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features, cropped = runner.get_features_from_image(img_rgb)
    # Perform inference
    res = None
    try:
        res = runner.classify(features)
    except Exception as e:
        print("ERROR: Could not perform inference")
        print("Exception:", e)
#    print("Output:", res)    # Display predictions and timing data

    if res is not None:
        # Find label with the highest probability
        predictions = res['result']['classification']
        max_label = ""
        max_val = 0
        for p in predictions:
            if predictions[p] > max_val:
                max_val = predictions[p]
                max_label = p
        
    return img, max_label, max_val



### Lay Flat ###
def general_lay_flat(img, view_edge):
    img, indices = edge_detect(img, view_edge)
    ## Find lowest point
    if np.any(indices):
        coords = list(zip(indices[1], indices[0]))
        max_y = np.argmax(indices[0])
        img = cv2.circle(img, coords[max_y], 10, (0,0,255), 2)
    return img



### Pants ###
def lay_flat_pants(img, view_edge):
    ## Grab waist
    img, tl, tr, bl, br = corner_detect(img, view_edge)
    cv2.circle(img, tl, 7, (0, 0, 255), -1)
    cv2.circle(img, tr, 7, (0, 0, 255), -1)
    return img

def pants_fold1(img, view_edge):
    ## Fold legs together
    img, tl, tr, bl, br = corner_detect(img, view_edge)
    cv2.circle(img, bl, 7, (0, 0, 255), -1)
    cv2.circle(img, tl, 7, (0, 0, 255), -1)
    cv2.circle(img, br, 7, (0, 0, 255), -1)
    cv2.circle(img, tr, 7, (0, 0, 255), -1)
    return img

def pants_fold2(img, view_edge):
    ## Fold into quarters
    img, tl, tr, bl, br = corner_detect(img, view_edge)
    cv2.circle(img, bl, 7, (0, 0, 255), -1)
    cv2.circle(img, tl, 7, (0, 0, 255), -1)
    mid = (round((tl[0]+bl[0])/2),round((tl[1]+bl[1])/2))
    q1 = (round((tl[0]+mid[0])/2),round((tl[1]+mid[1])/2))
    q3 = (round((mid[0]+bl[0])/2),round((mid[1]+bl[1])/2))
    cv2.circle(img, mid, 7, (0, 0, 255), -1)
    cv2.circle(img, q1, 7, (0, 0, 255), -1)
    cv2.circle(img, q3, 7, (0, 0, 255), -1)
    return img



### Long-Sleeve Shirts (LSS) ###
def lay_flat_lss(img, view_edge):
    ## Grab waist
    img, tl, tr, bl, br = corner_detect(img, view_edge)
    cv2.circle(img, tl, 7, (0, 0, 255), -1)
    cv2.circle(img, tr, 7, (0, 0, 255), -1)
    return img

def lss_fold1(img, view_edge):
    ## Grab midpoint of waist and collar
    img, indices = edge_detect(img, view_edge)
    if np.any(indices):
        coords = list(zip(indices[1], indices[0]))
        img_mid = int(img.shape[0]/2)
        pts = [t[1] for t in coords if t[0] == img_mid]
        for y in pts:
            img = cv2.circle(img, (img_mid, y), 10, (0,0,255), 2)
    return img

def lss_fold2(img, view_edge):
    ## Grab sleeves
    img, tl, tr, bl, br = corner_detect(img, view_edge)
    return img

def lss_fold3(img, view_edge):
    ## Fold into thirds
    img, tl, tr, bl, br = corner_detect(img, view_edge)
    return img
