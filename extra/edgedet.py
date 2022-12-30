import cv2
import numpy as np
 
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

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img = cv2.GaussianBlur(img, (7,7), 0) 
 
    # Canny Edge Detection
    img = cv2.Canny(img, 80, 200) # Canny Edge Detection
    indices = np.where(img == [255])  # Why are these backwards????
    if np.any(indices):
        coords = list(zip(indices[1], indices[0]))
        max_y = np.argmax(indices[0])
        max_x = np.argmax(indices[1])

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

    # Draw circles at POIs
    if np.any(indices):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.circle(img, coords[max_y], 10, (0,0,255), 2)
        img = cv2.circle(img, coords[max_x], 10, (0,0,255), 2)

    # Show the frame
    cv2.imshow('Imagetest',img)
    k = cv2.waitKey(1)
    if k != -1:
        break

# End
#cv2.imwrite('image1.jpg', image)
cam.release()
cv2.destroyAllWindows()
