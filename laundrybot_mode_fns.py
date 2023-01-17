import cv2
from edge_impulse_linux.image import ImageImpulseRunner
import numpy as np
import imutils


## Corner Detection
def corner_detect(img):
    # Convert to grayscale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Edge detection
    img = cv2.GaussianBlur(img, (7,7), 0) 
    img = cv2.Canny(img,80,200)

    # Corner Detection
    img = np.float32(img)
    dst = cv2.cornerHarris(img,30,13,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img[dst>0.1*dst.max()]=[0,0,255]

    # Blob Centroid Detection
    img_corners = np.zeros(img.shape, np.float32)
    img_corners = cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB)
    img_corners[dst>0.1*dst.max()]=[0,0,255]
    img_corners = np.uint8(img_corners)
    img_corners = cv2.cvtColor(img_corners, cv2.COLOR_BGR2GRAY)
    img_corners = cv2.GaussianBlur(img_corners, (5, 5), 0)
    img_corners = cv2.threshold(img_corners, 60, 255, cv2.THRESH_BINARY)[1]
    # find contours in the thresholded image
    cnts = cv2.findContours(img_corners.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bl = (img.shape[0], 0)
    tl = (img.shape[0], img.shape[1])
    br = (0, 0)
    tr = (0, img.shape[1])
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
        if (cX + (img.shape[1]-cY)) < (bl[0] + (img.shape[1]-bl[1])):
            bl = (cX, cY)
        if (cX + cY) < (tl[0] + tl[1]):
            tl = (cX, cY)
        if (cX + cY) > (br[0] + br[1]):
            br = (cX, cY)
        if ((img.shape[0]-cX) + cY) < (img.shape[0]-(tr[0]) + tr[1]):
            tr = (cX, cY)

    return tl, tr, bl, br



## Classify Image
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
    # Display predictions and timing data
#    print("Output:", res)

    # Display prediction on preview
    if res is not None:
        # Find label with the highest probability
        predictions = res['result']['classification']
        max_label = ""
        max_val = 0
        for p in predictions:
            if predictions[p] > max_val:
                max_val = predictions[p]
                max_label = p
        # Draw predicted label on bottom of preview
        cv2.putText(img,
                    max_label,
                    (0, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255,0,0))
        # Draw predicted class's confidence score (probability)
        cv2.putText(img,
                    str(round(max_val, 2)),
                    (0, img.shape[0] - 2),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255,0,0))
        
    return img



## Lay Flat (find lowest point)
def lay_flat(img, runner):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Edge Detection
    img = cv2.GaussianBlur(img, (7,7), 0) 
    img = cv2.Canny(img, 80, 200) # Canny Edge Detection
    indices = np.where(img == [255])  # Why are these backwards????
    if np.any(indices):
        coords = list(zip(indices[1], indices[0]))
        max_y = np.argmax(indices[0])

    # Draw circles at POIs
    if np.any(indices):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.circle(img, coords[max_y], 10, (0,0,255), 2)

    return img



## Pants
def pants_fold1_legs_togthr(img, runner):
    tl, tr, bl, br = corner_detect(img)
    cv2.circle(img, bl, 7, (0, 0, 255), -1)
    cv2.circle(img, tl, 7, (0, 0, 255), -1)
    cv2.circle(img, br, 7, (0, 0, 255), -1)
    cv2.circle(img, tr, 7, (0, 0, 255), -1)
    return img

def pants_fold2_quarters(img, runner):
    tl, tr, bl, br = corner_detect(img)
    cv2.circle(img, bl, 7, (0, 0, 255), -1)
    cv2.circle(img, tl, 7, (0, 0, 255), -1)
    mid = (round((tl[0]+bl[0])/2),round((tl[1]+bl[1])/2))
    q1 = (round((tl[0]+mid[0])/2),round((tl[1]+mid[1])/2))
    q3 = (round((mid[0]+bl[0])/2),round((mid[1]+bl[1])/2))
    cv2.circle(img, mid, 7, (0, 0, 255), -1)
    cv2.circle(img, q1, 7, (0, 0, 255), -1)
    cv2.circle(img, q3, 7, (0, 0, 255), -1)
    return img
