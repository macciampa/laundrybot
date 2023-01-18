from laundrybot_mode_fns import *
import os, sys, time
import cv2
from edge_impulse_linux.image import ImageImpulseRunner


### Initialization ###
def init_bot():
    # Impulse Runner Setup
    model_file = "modelfile.eim"             # Trained ML model from Edge Impulse
    dir_path   = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, model_file)
    runner     = ImageImpulseRunner(model_path)
    
    # Initialize model
    try:
        model_info = runner.init()
    except Exception as e:
        print("ERROR: Could not initialize model")
        print("Exception:", e)
        if (runner):
            runner.stop()
            sys.exit(1)
    
    # Setup Webcam
    cam = cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,360)

    return cam, runner



### Calculate and display Framerate ###
def disp_framerate(img, fps, timestamp):
    # Draw framerate on frame
    cv2.putText(img, 
                "FPS: " + str(round(fps, 2)), 
                (0, 12),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255,0,0))
    
    # Calculate framerate
    frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
    fps = 1 / frame_time
    return img, fps
