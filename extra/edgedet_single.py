import cv2
import numpy as np
 
# Read the original image
img = cv2.imread('pants.jpg') 
#cv2.imshow('Orig', img)
#cv2.waitKey(0)

# Convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img = cv2.GaussianBlur(img, (7,7), 0) 

# Canny Edge Detection
img = cv2.Canny(img,80,200)
indices = np.where(img == [255])
poi = np.argmax(indices[0])  # Why is this backwards????
coords = list(zip(indices[1], indices[0]))  # And this too???
#with open('coords.txt', 'w') as fp:
#    fp.write('\n'.join('{} {}'.format(x[0],x[1]) for x in coords))

# Draw circles at POI
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.circle(img, coords[poi], 10, (0,0,255), 2)

# Display Image
cv2.imshow('Edge Detection', img)
cv2.waitKey(0)
 
cv2.destroyAllWindows()
