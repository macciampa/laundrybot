#!/usr/bin/env python

import os, sys, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from edge_impulse_linux.image import ImageImpulseRunner

# Impulse Runner Setup
model_file = "modelfile.eim"             # Trained ML model from Edge Impulse
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, model_file)
runner = ImageImpulseRunner(model_path)

# Initialize model (and print information if it loads)
try:
    model_info = runner.init()
    print("Model name:", model_info['project']['name'])
    print("Model owner:", model_info['project']['owner'])
    
# Exit if we cannot initialize the model
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
res_height = 320                         # Resolution of camera (height)

# Initial framerate value
fps = 0

# Start webcam
while True:
    
    # Get timestamp for calculating actual framerate
    timestamp = cv2.getTickCount()

    ret, image = cam.read()

    # Crop
    img = image[20:340, 160:480]

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
    print("Output:", res)

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
                    (0, res_height - 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255,0,0))
                    
        # Draw predicted class's confidence score (probability)
        cv2.putText(img,
                    str(round(max_val, 2)),
                    (0, res_height - 2),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255,0,0))
        
    # Draw framerate on frame
    cv2.putText(img, 
                "FPS: " + str(round(fps, 2)), 
                (0, 12),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255,0,0))
    
    # Calculate framrate
    frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
    fps = 1 / frame_time

    # Show the frame
    cv2.imshow('Imagetest',img)
    k = cv2.waitKey(1)
    if k != -1:
        break

# End
#cv2.imwrite('image1.jpg', image)
cam.release()
cv2.destroyAllWindows()
