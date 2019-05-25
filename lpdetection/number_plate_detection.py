import cv2
import numpy as np


def increase_contrast(bgr_img):
    """
    Increase the contrast of the input image
    :param bgr_img: cv2 read image
    :return: the image with an increased contrast
    """
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


def areOverlapping(rect1, rect2):
    """
    Find if the two rectangles are intersecting.
    Rect: (topLeft(x,y), bottomRight(x,y))
    """
    topLeft1 = rect1[0]
    bottomRight1 = rect1[1]

    topLeft2 = rect2[0]
    bottomRight2 = rect2[1]

    if topLeft1[0] > bottomRight2[0] or bottomRight1[0] < topLeft2[0]:
        return False
    if topLeft1[1] > bottomRight2[1] or bottomRight1[1] < topLeft2[1]:
        return False
    return True


def findRectangleIntersections(rectangles):
    """
    Find the intersection of every rectangles

    :param rectangles: a list containing tuples of points (topLeft, bottomRight)
    :return: the filtered list of points
    """
    result = rectangles.copy()
    intersected = True
    i = 0
    j = 1
    while intersected:
        intersected = False

        if len(result) >= 2 and areOverlapping(result[i], result[j]):
            minRectangle = minimumContainingRect(rectangles[i], rectangles[j])
            result.append(minRectangle)
            result.remove(rectangles[i])
            result.remove(rectangles[j])
            i = 0
            j = 1
            intersected = True
        else:
            j += 1
            if j == len(result):
                j = 0
                i += 1
            if i == len(result) - 1:
                break
    return result


def minimumContainingRect(rect1, rect2):
    """
    Rect: (topLeft(x,y), bottomRight(x,y))
    :return: the minimum rectangle that contains both of the rectangles.
    """
    intTopLeft = (min(rect1[0][0], rect1[0][0]), min(rect2[0][1], rect2[0][1]))
    intBottomRight = (max(rect1[1][0], rect1[1][0]), max(rect2[1][1], rect2[1][1]))
    return intTopLeft, intBottomRight


def filterHorizontalRectangles(rectangles):
    """
    Filter the horizontal oriented rectangles from the given list.

    :param rectangles: a list containing tuples of points (topLeft, bottomRight)
    :return: the filtered list of points
    """
    result = []
    for (topLeft, bottomRight) in rectangles:
        rectangleWidth = bottomRight[0] - topLeft[0]
        rectangleHeight = bottomRight[1] - topLeft[1]
        if rectangleWidth > rectangleHeight:
            result.append((topLeft, bottomRight))
    return result


class NumberPlateDetection:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def detect_number_plate_locations(input_img):
        """
        Detect the number plate locations from the input image.

        :param input_img: cv2 read input image
        :return:  all the locations cropped as cv2 images from the input image
        """
        # BLUR the image
        blurred = cv2.GaussianBlur(input_img, (3, 3), 0)

        # Increase image CONTRAST
        contrasted = increase_contrast(blurred)

        # Convert the image to grayscale
        gray = cv2.cvtColor(contrasted, cv2.COLOR_BGR2GRAY)

        # Canny edges
        edges = cv2.Canny(gray, 100, 200)

        # Find Contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        img_height = input_img.shape[0]
        img_width = input_img.shape[1]

        number_rectangles = []
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
                topLeft, bottomRight = find_margin_corners(x, y, box)
                number_rectangles.append((topLeft, bottomRight))

        # If there are intersecting rectangles, replace them with the minimum containing rectangle
        filteredRectangles = findRectangleIntersections(number_rectangles)
        # From the rectangles, only keep the ones that have a horizontal orientation
        filteredRectangles = filterHorizontalRectangles(filteredRectangles)

        print('{0:d} numbers found'.format(len(filteredRectangles)))

        number_images = []
        for (top_left, bottom_right) in filteredRectangles:
            number_img = input_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            number_images.append(number_img)

        return number_images
