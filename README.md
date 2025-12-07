# assignment-1
1)Difference between cv2.waitKey(0) and cv2.waitKey(1)
cv2.waitKey(delay) waits for a keyboard event for delay milliseconds and returns the pressed key (as an integer) or -1 if no key was pressed.

3)
img_bgr = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

4)
import cv2
import numpy as np

img = cv2.imread("image.jpg")

choice = input("darken / lighten / invert: ")

if choice == "darken":
    out = img * 0.5
elif choice == "lighten":
    out = img * 1.5
elif choice == "invert":
    out = 255 - img
else:
    out = img

out = out.astype("uint8")
cv2.imshow("result", out)
cv2.waitKey(0)

4)
import cv2
import numpy as np

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

min_v = 100
max_v = 150

mask = cv2.inRange(gray, min_v, max_v)
out = img.copy()
out[mask == 255] = [255, 255, 255]

cv2.imshow("result", out)
cv2.waitKey(0)

5)
import cv2
import numpy as np

img = cv2.imread("image.jpg")

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], np.float32)

out = cv2.filter2D(img, -1, kernel)

cv2.imshow("filtered", out)
cv2.waitKey(0)

6)
import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)

g1 = cv2.GaussianBlur(img, (0,0), 1)
g2 = cv2.GaussianBlur(img, (0,0), 2)

dog = g1 - g2

cv2.imshow("DoG", dog)
cv2.waitKey(0)

