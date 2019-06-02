import cv2

from textdetection import text_recognition
from visionapi import vision
from lpdetection import number_plate_detection
from yolov3 import car_detection
from utils import text_filter
from collections import defaultdict

NO_DATE = "NO_DATE"

cap = cv2.VideoCapture("input/cctv1.mp4")
if not cap.isOpened():
    print("Error opening the video file. Please double check your file path for typos. "
          "Or move the movie file to the same location as this script/notebook")

FPS = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print(str(FPS) + " frames per second.")
print(str(frame_count) + " frames read from the file.")
print(str(frame_height) + " frames height.")
print(str(frame_width) + " frames width.")

frame_number = 0
frames = []
while cap.isOpened():
    # Read the video file frame by frame.
    ret, frame = cap.read()

    # Work with two consecutive frames from every second of the video.
    if ret:
        frame_number += 1
        if frame_number % FPS == 0 or frame_number % FPS == 1:
            frames.append(frame)
    else:
        break

print(str(len(frames)) + " frames collected for the recognition.")

# Detectors
eastDetector = text_recognition.EastTextDetector()
yoloDetector = car_detection.YoloDetector()
nplDetector = number_plate_detection.NumberPlateDetection()
nprTextsFilter = text_filter.NprTextsFilter()
visionDetector = vision.Vision()


def get_date_from_margins(input_image, visionDetector, nprTextsFilter):
    """
    Extract the date from the margins of the image (top left & right or bottom left & right).

    :param nprTextsFilter: the text filter, used to keep only the valid dates
    :param visionDetector: the Vision API text detector
    :param input_image: input cv2 image
    :return: the date found from the first detection from within the image margins or None if note existent
    """
    image = input_image.copy()
    height, width = image.shape[:2]

    margin_corners = [
        # top left margin
        image[0:int(height * 0.1), 0:int(width * 0.5)],
        # top right margin corner
        image[0:int(height * 0.1), int(width * 0.5):width],
        # bottom left margin
        image[int(height * 0.9): height, 0:int(width * 0.5)],
        # bottom right margin
        image[int(height * 0.9): height, int(width * 0.5):width]]

    for margin in margin_corners:
        texts = visionDetector.detect_texts(margin)
        dates = nprTextsFilter.filterDates(texts)
        if len(dates) > 0:
            return dates[0]
    return None


result = defaultdict(list)

for frame in frames:

    # Initial text recognition using east text detection and recognition
    east_date, east_numbers = eastDetector.extract_numbers_first_date(frame)

    # Car detection from within every frame
    detected_vehicles = yoloDetector.detect_cars(frame)
    print("cars detected:" + str(len(detected_vehicles)))

    # Number Plate Location Detection for every detected vehicle
    detected_numbers = []
    for vehicle in detected_vehicles:
        # For every number plate location detected get the text from it using Vision API
        number_plates = nplDetector.detect_number_plate_locations(vehicle)

        for nr_plate in number_plates:
            # All detected text from the number plate location
            plate_texts = visionDetector.detect_texts(nr_plate)

            # collect the filtered romanian number plates
            ignore, romanian_plates = nprTextsFilter.filterDatesAndPlates(plate_texts)
            detected_numbers.extend(romanian_plates)

    # If we do not receive a date then we try to detect it from the 4 corners of the frame with Vision
    date = get_date_from_margins(frame, visionDetector, nprTextsFilter) \
        if east_date is None and len(detected_numbers) > 0 else east_date

    map_key = date if date is not None else NO_DATE
    distinct_numbers = set(detected_numbers)
    for number in distinct_numbers:
        if number not in result[map_key]:
            result[map_key].append(number)

print(result)
