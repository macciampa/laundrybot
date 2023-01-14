import os, sys, time
import cv2
from edge_impulse_linux.image import ImageImpulseRunner


## Initialization
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

    # Create dispatch table for modes
    modes = {
        "classify": img_classification
    }

    return cam, runner, modes



## Calculate and display Framerate
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
