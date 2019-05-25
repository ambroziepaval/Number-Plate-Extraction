from visionapi import vision
from lpdetection import number_plate_detection
from yolov3 import car_detection
import cv2

test_frame = cv2.imread("visionapi/test_frame.png")
test_car = cv2.imread("lpdetection/car.jpg")

# Integration Test: Vision API
# vision_api = vision.Vision()
# texts = vision_api.detect_texts(test_frame)
# print(texts)

# Integration Test: Number Plate Detection
# np_detection = number_plate_detection.NumberPlateDetection()
# number_images = np_detection.detect_number_plate_locations(test_car)
# print(len(number_images))

# Integration Test: YOLOv3 detection
yoloDetector = car_detection.YoloDetector()
detected_cars = yoloDetector.detect_cars(test_frame)
print(len(detected_cars))
