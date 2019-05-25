import cv2
import numpy as np


def increase_contrast(bgr_img):
    # Convert image to LAB color space
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

    # Split the 3 channels
    l, a, b = cv2.split(lab_img)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the other channels
    l_img = cv2.merge((cl, a, b))

    final = cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)
    return final


def find_margin_corners(x, y, boxPoints):
    """ Find top left and bottom right corners of the rectangle
        (x, y) - center of the rectangle
        boxPoints - all 4 corner points
    """
    topLeft = None
    bottomRight = None

    for (px, py) in boxPoints:
        if px <= x and py <= y:
            topLeft = (px, py)
        if px >= x and py >= y:
            bottomRight = (px, py)
    return topLeft, bottomRight


### 1st variant ###
img = cv2.imread('car1.jpg')
# display_bgr(img)

# BLUR the image
blurred = cv2.GaussianBlur(img, (3, 3), 0)
# display_bgr(blurred)

# Increase CONTRAST
contrasted = increase_contrast(blurred)
# display_bgr(contrasted)

# Gray
gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)
# display_gray(gray)

# Canny edges
edges = cv2.Canny(gray, 100, 200)
# display_gray(edges)

# Find Contours
closed = edges.astype(np.uint8)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
external_contours = img.copy()
for i in range(len(contours)):
    # only external contours (-1)
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours, i, (0, 0, 255), 1)
# display_bgr(external_contours)

# Min Area Rects
img_copy = img.copy()
for i in range(len(contours)):
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 1)
# display_bgr(img_copy)

###### Filtered Area Rects by angle ######
print('Filtered Area Rects by angle')
img_copy = img.copy()
for i in range(len(contours)):
    rect = cv2.minAreaRect(contours[i])
    (x, y), (width, height), angle = rect

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if abs(rect[2]) >= 80 and abs(rect[2]) <= 100:
        #         print (x ,y, width, height, angle)
        cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 1)
# display_bgr(img_copy)

###### Filtered Area Rects by angle and width + height ######
# print('Filtered Area Rects by angle and width + height')
img_copy = img.copy()
img_height = img.shape[0]
img_width = img.shape[1]
count = 0
for i in range(len(contours)):
    rect = cv2.minAreaRect(contours[i])
    (x, y), (width, height), angle = rect

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = float(width)
    height = float(height)
    rightRatio = False
    if width > 0.0 and height > 0.0:
        rightRatio = (3.0 < width / height < 5.0) or (3.0 < height / width < 5.0)

    if ((80 <= abs(angle) <= 100) or (abs(angle) <= 10)) and rightRatio \
            and max(width, height) > 50 and min(width, height) > 10 \
            and img_height / height > 5 and img_width / width > 3:
        #         print(box)
        cv2.circle(img_copy, (box[0][0], box[0][1]), 2, (0, 255, 0), -1)
        cv2.circle(img_copy, (box[1][0], box[1][1]), 2, (255, 0, 0), -1)
        cv2.circle(img_copy, (box[2][0], box[2][1]), 2, (0, 0, 255), -1)
        cv2.circle(img_copy, (box[3][0], box[3][1]), 2, (200, 200, 0), -1)

        cv2.circle(img_copy, (int(x), int(y)), 1, (255, 255, 0), -1)
        cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 1)

        topLeft, bottomRight = find_margin_corners(x, y, box)
        number_img = img[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        cv2.imwrite('numbers/number_' + str(count) + '.jpg', number_img)
        count += 1
        print('{0:d} numbers found'.format(count))

# display_bgr(img_copy)
