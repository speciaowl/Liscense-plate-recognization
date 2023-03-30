import cv2
import imutils

image = cv2.imread("car.jpeg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the binary image
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Loop over the contours
for contour in contours:
    # Approximate the contour
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # If the contour has four vertices, it could be a number plate
    if len(approx) == 4:
        # Get the bounding box coordinates of the contour
        x, y, w, h = cv2.boundingRect(approx)
        
        aspect_ratio = w / float(h)
        if 2 <= aspect_ratio <= 6 and w > 100 and h > 20:
            # Draw the contour on the image
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

cv2.imshow("Number Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
