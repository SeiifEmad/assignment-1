import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

mode = 'o'      # Default display mode
sigma = 1.0     # Smoothing parameter (Gaussian blur)

print("\nControls:")
print(" o - Original frame")
print(" x - Sobel X")
print(" y - Sobel Y")
print(" m - Sobel Magnitude")
print(" s - Sobel + Threshold")
print(" l - Laplacian of Gaussian (LoG)")
print(" + - Increase sigma")
print(" - - Decrease sigma")
print(" q - Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Always blur before applying gradients / edges
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)

    # ORIGINAL
    if mode == 'o':
        output = frame.copy()
        text = "Original"

    # SOBEL X
    elif mode == 'x':
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        output = cv2.convertScaleAbs(sobel_x)
        text = "Sobel X"

    # SOBEL Y
    elif mode == 'y':
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        output = cv2.convertScaleAbs(sobel_y)
        text = "Sobel Y"

    # SOBEL MAGNITUDE
    elif mode == 'm':
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        output = cv2.convertScaleAbs(magnitude)
        text = "Sobel Magnitude"

    # SOBEL + THRESHOLD (bonus)
    elif mode == 's':
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = cv2.convertScaleAbs(magnitude)
        _, output = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
        text = "Sobel + Threshold"

    # LAPLACIAN OF GAUSSIAN (LoG)
    elif mode == 'l':
        # First blur with given sigma
        blurred_sigma = cv2.GaussianBlur(gray, (0, 0), sigma)
        # Then apply Laplacian
        log = cv2.Laplacian(blurred_sigma, cv2.CV_64F)
        output = cv2.convertScaleAbs(log)
        text = f"LoG (sigma={sigma:.1f})"

    # Display mode text on frame
    cv2.putText(output, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Webcam Edge Detection", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord('o'), ord('x'), ord('y'), ord('m'), ord('s'), ord('l')]:
        mode = chr(key)
    elif key == ord('+'):
        sigma = min(10, sigma + 0.2)
    elif key == ord('-'):
        sigma = max(0.2, sigma - 0.2)

cap.release()
cv2.destroyAllWindows()
