#!/usr/bin/env python
from laundrybot_fns import *
import cv2

## Initialize model and webcam
cam, runner, modes = init_bot()

## Start webcam
fps = 0
curr_mode = "classify"

while True:
    timestamp = cv2.getTickCount()   # Get timestamp for calculating framerate
    ret, img = cam.read()            # Grab from webcam
    img = img[20:340, 160:480]       # Crop

    # State Machine
    runfn = modes[curr_mode]
    img = runfn(img, runner)   # Perform classification

    # Show the frame
    img, fps = disp_framerate(img, fps, timestamp)   # Calculate and display framerate
    cv2.imshow('LaundryBotCam',img)

    # Determine next state
    match cv2.waitKey(1):
        case 113 | 32: # q or Spacebar
            break
        case _:
            continue

# End
cam.release()
cv2.destroyAllWindows()
