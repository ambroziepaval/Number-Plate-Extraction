import time

import os
import cv2
import numpy as np

from yolov3.model.yolo_model import YOLO


def process_image(img):
    """ Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """ Get classes names for the YOLO detection.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.
    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """

    count = 0
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        if all_classes[cl] == 'car' or all_classes[cl] == 'bus':
            car_img = image[left:bottom, top:right]
            cv2.imwrite('car_' + str(count) + '.jpg', car_img)
            count += 1
            print('{0:d} cars found'.format(count))

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))


def extract_cars(image, boxes, scores, classes, all_classes):
    """ Extract the detected cars & buses from the image.

    :param image: original image
    :param boxes: ndarray, boxes of objects
    :param scores: ndarray, scores of objects.
    :param classes: ndarray, classes of objects.
    :param all_classes: all classes name.
    :return: the list of all images with detected cars/buses
    """

    cars = []

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        if all_classes[cl] == 'car' or all_classes[cl] == 'bus':
            car_img = image[left:bottom, top:right]
            cars.append(car_img)

    return cars


def detect_image(image, yolo, all_classes):
    """ Use yolo v3 to detect objects in the images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_cars_image(image, yolo, all_classes):
    """
    Use yolo v3 to detect cars / buses within the given image.

    :param image: image to detect from
    :param yolo: the yolo model
    :param all_classes: all classes from yolo
    :return:
    """
    processed_image = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(processed_image, image.shape)
    end = time.time()

    print('Detection time: {0:.2f}s'.format(end - start))

    cars = []
    if boxes is not None:
        cars = extract_cars(image, boxes, scores, classes, all_classes)

    return cars


def testYoloDetection():
    # load the YOLO model
    yolo = YOLO(0.6, 0.5)
    # load the YOLO available classes
    all_classes = get_classes('data/coco_classes.txt')
    img = 'test_frame.png'
    path = 'images/' + img
    image = cv2.imread(path)
    # img = detect_image(image, yolo, all_classes)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    detected_cars = detect_cars_image(image, yolo, all_classes)
    print('Cars detected: ' + str(len(detected_cars)))
    for (i, car) in enumerate(detected_cars):
        path = 'images/cars/car' + str(i) + '.jpg'
        cv2.imwrite(path, car)


# testYoloDetection()


class YoloDetector:
    __yolo = None
    __all_classes = None

    def __init__(self) -> None:
        # load the YOLO model
        self.yolo = YOLO(0.6, 0.5)
        # load the YOLO available classes
        self.all_classes = get_classes(os.path.dirname(__file__) + '/data/coco_classes.txt')

    def detect_cars(self, image):
        detected_cars = detect_cars_image(image, self.yolo, self.all_classes)
        return detected_cars
