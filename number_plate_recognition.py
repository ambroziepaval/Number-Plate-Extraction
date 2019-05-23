import cv2

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

