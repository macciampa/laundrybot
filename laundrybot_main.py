#!/usr/bin/env python
from laundrybot_basic_fns import *
from laundrybot_mode_fns import *
import cv2
import numpy as np

## Initialize model and webcam
cam, runner = init_bot()
aa_size     = 5    # size of average array for classification
conf_ratio  = 0.4  # confidence ratio for classification
fps         = 0
curr_mode   = "lss_fold1"
view_edge   = False  # view edge detect instead of normal image

while True:
    timestamp = cv2.getTickCount()   # Get timestamp for calculating framerate
    ret, img = cam.read()            # Grab from webcam
    img = img[20:340, 160:480]       # Crop

    # State Machine
    match curr_mode:
        case "lay_flat":
            img = general_lay_flat(img, view_edge)
            next_mode  = "classify"
            avg_arr    = aa_size*[0]
            last_label = ""
        case "classify":
            img, label, val = img_classification(img, runner)
            cv2.putText(img,label,(0, img.shape[0]-20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0))            # Draw predicted label on bottom of preview
            cv2.putText(img,str(round(val,2)),(0,img.shape[0]-2),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0))  # Draw predicted class's confidence score (probability)
            if label == last_label:
                avg_arr = np.append(avg_arr[1:aa_size],[val])
                if np.average(avg_arr) > conf_ratio:
                    match label:
                        case 'pants':
                            next_mode = "lay_flat_pants"
                        case 'lss':
                            next_mode = "lay_flat_lss"
                else:
                    next_mode = "classify"
            else:
                avg_arr = aa_size*[0]
                next_mode = "classify"
            last_label = label
        case "lay_flat_pants":
            img = lay_flat_pants(img, view_edge)
            next_mode = "pants_fold1"
        case "pants_fold1":
            img = pants_fold1(img, view_edge)
            next_mode = "pants_fold2"
        case "pants_fold2":
            img = pants_fold2(img, view_edge)
            next_mode = "lay_flat"
        case "lay_flat_lss":
            img = lay_flat_lss(img, view_edge)
            next_mode = "lss_fold1"
        case "lss_fold1":
            img = lss_fold1(img, view_edge)
            next_mode = "lss_fold2"
        case "lss_fold2":
            img = lss_fold2(img, view_edge)
            next_mode = "lss_fold3"
        case "lss_fold3":
            img = lss_fold3(img, view_edge)
            next_mode = "lay_flat"

    # Show the frame
    img, fps = disp_framerate(img, fps, timestamp)   # Calculate and display framerate
    cv2.imshow('LaundryBotCam',img)
    cv2.setWindowTitle('LaundryBotCam',curr_mode)

    # Check for user input
    match cv2.waitKey(1):
        case 32: # Spacebar
            curr_mode = next_mode
        case 101: # e
            view_edge = not view_edge
        case 113: # q
            break


## End
cam.release()
cv2.destroyAllWindows()
