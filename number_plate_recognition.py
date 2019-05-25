import cv2

from visionapi import vision
from lpdetection import number_plate_detection
from yolov3 import car_detection

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
yoloDetector = car_detection.YoloDetector()
nplDetector = number_plate_detection.NumberPlateDetection()
visionDetector = vision.Vision()

# TODO change to get all the frames
for frame in frames[0:2]:
    # TODO extract the text from the frame, using east text detector (possible dates / numbers)
    # If we do not receive a date then we try to detect it from the 4 corners of the frame with Vision

    # Car detection from within every frame
    detected_vehicles = yoloDetector.detect_cars(frame)
    print("cars detected:" + str(len(detected_vehicles)))

    # Number Plate Location Detection for every detected vehicle
    for vehicle in detected_vehicles:
        # For every number plate location detected get the text from it using Vision API
        number_plates = nplDetector.detect_number_plate_locations(vehicle)

        for nr_plate in number_plates:
            # All detected text from the number plate location
            plate_texts = visionDetector.detect_texts(nr_plate)

    # TODO parse every detected text and check if it is a date or number (create a custom filter)
    # TODO if the date was not detected at the initial step try and detect it with Vision API
