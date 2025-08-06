import numpy as np
import cv2

# Create a blank white image
img = np.full((60, 60), 255, dtype=np.uint8)

# Draw a black letter
cv2.putText(img, '3', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0), 3)

cv2.imwrite('number_3.png', img)
