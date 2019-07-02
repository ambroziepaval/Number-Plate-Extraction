import cv2

from collections import defaultdict

from lpdetection import number_plate_detection
from textdetection import text_recognition
from utils import text_filter
from visionapi import vision
from yolov3 import car_detection


def write_result(result_map):
    """
    Write the result map into a txt file

    :param result_map: the detection result map Map <Date, List<Number>>
    """
    global date, number
    file = open("result.txt", "w")
    for date in result_map.keys():
        for number in result_map[date]:
            file.write(str(date) + " - " + number)
    file.close()


NO_DATE = "NO_DATE"

# Detectors
eastDetector = text_recognition.EastTextDetector()
yoloDetector = car_detection.YoloDetector()
nplDetector = number_plate_detection.NumberPlateDetection()
visionDetector = vision.Vision()
nprTextsFilter = text_filter.NprTextsFilter()


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
input_image = cv2.imread("input/image.png")

# Initial text recognition using east text detection and recognition
east_date, east_numbers = eastDetector.extract_numbers_first_date(input_image)

# Car detection from the input image
detected_vehicles = yoloDetector.detect_cars(input_image)
print("cars detected:" + str(len(detected_vehicles)))

# Number Plate Location Detection for every detected vehicle
detected_numbers = []
for vehicle in detected_vehicles:
    # print("Vehicle " + str(detected_vehicles.index(vehicle) + 1))
    # For every number plate location detected get the text from it using Vision API
    number_plates = nplDetector.detect_number_plate_locations(vehicle)

    for nr_plate in number_plates:
        # All detected text from the number plate location
        plate_texts = visionDetector.detect_texts(nr_plate)

        # collect the filtered romanian number plates
        ignore, romanian_plates = nprTextsFilter.filterDatesAndPlates(plate_texts)
        detected_numbers.extend(romanian_plates)

# If we do not receive a date then we try to detect it from the 4 corners of the image with Vision
if east_date is None:
    print("EAST Date recognition failed...")
    print("Trying Vision API Date Recognition...")
date = get_date_from_margins(image, visionDetector, nprTextsFilter) \
    if east_date is None and len(detected_numbers) > 0 else east_date

# add the detected number plates into a Map <Date, List<Number>>
map_key = date if date is not None else NO_DATE
distinct_numbers = set(detected_numbers)
for number in distinct_numbers:
    if number not in result[map_key]:
        result[map_key].append(number)

print(result)
write_result(result)
