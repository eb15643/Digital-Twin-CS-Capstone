import cv2
import numpy as np
import signal

from CameraTransformation import createTopDownView, computeTransformationMatrix

# Define a flag to control the while loop
running = True

# Signal handler function
def signal_handler(sig, frame):
    global running
    print('Stopping gracefully...')
    running = False


# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def resizeFrameToMatch(frame1, frame2):
    height1, width1 = frame1.shape[:2]
    height2, width2 = frame2.shape[:2]
    if height1 < height2:
        scaling_factor = height2 / float(height1)
        frame1 = cv2.resize(frame1, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    elif height1 > height2:
        scaling_factor = height1 / float(height2)
        frame2 = cv2.resize(frame2, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame1, frame2

def stitchFrames(frame1, frame2):
    frame1, frame2 = resizeFrameToMatch(frame1, frame2)
    return np.hstack((frame1, frame2))

# Placeholder for RTSP URLs
rtsp_url1 = "rtsp://141.165.40.33/stream1"
rtsp_url2 = "rtsp://141.165.40.34/stream1"

# Initialize video capture for the RTSP streams
cap1 = cv2.VideoCapture(rtsp_url1)
cap2 = cv2.VideoCapture(rtsp_url2)

# Assume transformation points are predefined for both cameras
points1 = [(1265, 1047), (1579, 333), (865, 91), (413, 513)]
points2 = [(1290, 982), (1526, 354), (979, 297), (596, 712)]

# Compute transformation matrices for both streams
matrix1, new_width1, new_height1 = computeTransformationMatrix(points1, rotate=True)
matrix2, new_width2, new_height2 = computeTransformationMatrix(points2, rotate=False)

while running:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Failed to grab frames.")
        break

    transformed_frame1 = createTopDownView(frame1, matrix1, new_width1, new_height1, rotate=True)
    transformed_frame2 = createTopDownView(frame2, matrix2, new_width2, new_height2)

    stitched_frame = stitchFrames(transformed_frame2, transformed_frame1)

    # Display the resulting frame
    cv2.imshow('Stitched Frame', stitched_frame)

    # Check for 'q' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap1.release()
cap2.release()
cv2.destroyAllWindows()
