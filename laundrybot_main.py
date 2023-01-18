#!/usr/bin/env python
from laundrybot_basic_fns import *
from laundrybot_mode_fns import *
import cv2

## Initialize model and webcam
cam, runner = init_bot()

## Start webcam
fps = 0
curr_mode = "lay_flat"

while True:
    timestamp = cv2.getTickCount()   # Get timestamp for calculating framerate
    ret, img = cam.read()            # Grab from webcam
    img = img[20:340, 160:480]       # Crop

    # State Machine
    match curr_mode:
        case "lay_flat":
            img = lay_flat(img)
            next_mode = "classify"
            avg_arr = 10*[0]
        case "classify":
            img = img_classification(img, runner)
            next_mode = "pants_lay_flat"
        case "pants_lay_flat":
            img = pants_lay_flat(img)
            next_mode = "pants_fold1"
        case "pants_fold1":
            img = pants_fold1(img)
            next_mode = "pants_fold2"
        case "pants_fold2":
            img = pants_fold2(img)
            next_mode = "lay_flat"

    # Show the frame
    img, fps = disp_framerate(img, fps, timestamp)   # Calculate and display framerate
    cv2.imshow('LaundryBotCam',img)

    # Check for user input
    match cv2.waitKey(1):
        case 32: # Spacebar
            curr_mode = next_mode
        case 113: # q
            break

# End
cam.release()
cv2.destroyAllWindows()
