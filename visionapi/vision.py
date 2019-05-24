import io
import os
import cv2
import base64
import numpy as np
from google.cloud import vision
from google.cloud.vision import types


class Vision:
    client = None

    def __init__(self) -> None:
        self.client = vision.ImageAnnotatorClient()

    def detect_texts(self, input_img):
        """
        Detect all the text from a ccv2 image using the Google Cloud Vision API.

        :param input_img: cv2 loaded image
        :return: the list of all detected texts
        """
        retval, buffer = cv2.imencode('.jpg', input_img)
        img_bytes = np.array(buffer).tobytes()

        image = types.Image(content=img_bytes)

        response = self.client.text_detection(image=image)
        detected_texts = response.text_annotations

        texts = []
        for text in detected_texts:
            print('\n"{}"'.format(text.description))
            texts.append(text.description)

        return texts
