from visionapi import vision
from lpdetection import number_plate_detection
import cv2

vision_api = vision.Vision()
np_detection = number_plate_detection.NumberPlateDetection()

# Integration Test: Vision API
# texts = vision_api.detect_texts(cv2.imread("visionapi/test_frame.png"))
# print(texts)

# Integration Test: Number Plate Detection
car_img = cv2.imread("lpdetection/car.jpg")

number_images = np_detection.detect_number_plate_locations(car_img)
print(len(number_images))

for image in number_images:
    cv2.imshow('test', image)
    cv2.waitKey(0)

