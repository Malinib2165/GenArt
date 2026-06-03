import cv2
import numpy as np

# Create a simple test image
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
cv2.imwrite('test_image.png', img)
print("Test image created: test_image.png")
