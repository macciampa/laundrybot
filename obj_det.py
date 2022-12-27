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

# Initial framerate value
fps = 0

# Start webcam
while True:
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
        
        # Go through each of the returned bounding boxes
        bboxes = res['result']['bounding_boxes']
        for bbox in bboxes:
        
            # Calculate corners of bounding box so we can draw it
            b_x0 = bbox['x']
            b_y0 = bbox['y']
            b_x1 = bbox['x'] + bbox['width']
            b_y1 = bbox['y'] + bbox['height']
            
            # Draw bounding box over detected object
            cv2.rectangle(img,
                            (b_x0, b_y0),
                            (b_x1, b_y1),
                            (255, 255, 255),
                            1)
                            
            # Draw object and score in bounding box corner
            cv2.putText(img,
                        bbox['label'] + ": " + str(round(bbox['value'], 2)),
                        (b_x0, b_y0 + 12),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255))
        
        # Draw framerate on frame
        cv2.putText(img, 
                    "FPS: " + str(round(fps, 2)), 
                    (0, 12),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255))
        
        # Show the frame
        cv2.imshow('Imagetest',img)
        k = cv2.waitKey(1)
        if k != -1:
                break

# End
#cv2.imwrite('image1.jpg', image)
cam.release()
cv2.destroyAllWindows()
