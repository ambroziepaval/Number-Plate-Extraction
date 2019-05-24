from visionapi import vision
import cv2

vision_api = vision.Vision()

texts = vision_api.detect_texts(cv2.imread("visionapi/test_frame.png"))
print(texts)
