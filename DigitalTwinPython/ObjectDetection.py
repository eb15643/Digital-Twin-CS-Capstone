import cv2
import torch
import os
import sys
import numpy as np

# Import specific classes and functions for YOLO object detection
from YOLOv5.models.common import DetectMultiBackend
from YOLOv5.utils.general import non_max_suppression, scale_boxes
from YOLOv5.utils.torch_utils import select_device
from YOLOv5.utils.dataloaders import letterbox

# Leave this. Import is used when accessing RTSP camera feeds for detection
from CameraTransformation import createTopDownView, computeTransformationMatrix

# Set up paths for YOLOv5 integration from local file
currentUrl = os.path.dirname(__file__)
yolo5_path = os.path.abspath(os.path.join(currentUrl, 'YOLOv5'))
sys.path.append(yolo5_path)

# Initialize the device (CPU or GPU) and load the YOLO model with custom trained weights
device = select_device('cpu')  # Replace 'cpu' with 'cuda' to use GPU
weights = 'YOLOv5/runs/train/exp/weights/best.pt'  # Path to the model's weight file
model = DetectMultiBackend(weights, device=device)
stride, names, _ = model.stride, model.names, model.pt
img_size = 640  # Define the input size for the model

# A very rudimentary tracker for consistent IDs for objects across frames
class SimpleTracker:
    def __init__(self, threshold=0.3, timeout=300):
        self.objects = {}
        self.next_id = 1
        self.threshold = threshold
        self.timeout = timeout  # frames to keep an ID without detection

    def update_objects(self, detected_objects, current_frame):
        new_objects = {}
        # Temporarily keep old objects that weren't updated in this frame
        for obj_id, (obj_label, (old_x, old_y), last_seen) in self.objects.items():
            if current_frame - last_seen <= self.timeout:
                new_objects[obj_id] = (obj_label, (old_x, old_y), last_seen)

        # Check each detected object against existing ones
        for label, (x, y) in detected_objects:
            found = False
            for obj_id, (obj_label, (old_x, old_y), last_seen) in new_objects.items():
                if label == obj_label and abs(x - old_x) <= self.threshold and abs(y - old_y) <= self.threshold:
                    # Update existing object's coordinates and last seen frame count
                    new_objects[obj_id] = (label, (x, y), current_frame)
                    found = True
                    break
            if not found:
                # Assign a new ID to a new object with current frame as last seen
                new_objects[self.next_id] = (label, (x, y), current_frame)
                self.next_id += 1

        # Update the tracker's list to only include updated or still valid objects
        self.objects = new_objects

    def get_objects(self):
        # Return objects without the frame count
        return {obj_id: (obj_label, (x, y)) for obj_id, (obj_label, (x, y), last_seen) in self.objects.items()}


# For conversion of detected coordinates to Unreal
class CoordinateMapper:
    def __init__(self):
        # Coefficients for x' and y' calculations
        self.coeff_x = [5.53537811e-03, 1.12625603e-04, -0.129592638]
        self.coeff_y = [4.62364054, -0.0004116201, -0.0050816166,
                        1.58095320e-08, 2.53286199e-07, 7.80706109e-07]

    def convert(self, x, y):
        # Calculate new x using the linear coefficients
        x_new = self.coeff_x[0] * x + self.coeff_x[1] * y + self.coeff_x[2]

        # Calculate new y using polynomial coefficients (degree 2)
        y_new = (self.coeff_y[0] +
                 self.coeff_y[1] * x +
                 self.coeff_y[2] * y +
                 self.coeff_y[3] * x**2 +
                 self.coeff_y[4] * x * y +
                 self.coeff_y[5] * y**2)

        return x_new, y_new


def detect(frame, frame_count, tracker):
    img = letterbox(frame, img_size, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))  # Convert HWC to CHW
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # Normalize to 0-1
    if len(img.shape) == 3:
        img = img[None]  # Add a batch dimension

    pred = model(img, augment=False)  # Run detection
    pred = non_max_suppression(pred, 0.5, 0.45, max_det=1000)  # Apply NMS
    mapper = CoordinateMapper()

    detected_objects = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()  # Adjust for original frame size
            for *xyxy, conf, cls in reversed(det):
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])  # Bottom-right corner for the bounding box
                converted_x, converted_y = mapper.convert(x1, y1)
                label = f"{names[int(cls)]}"
                detected_objects.append((label, (converted_x, converted_y)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Update tracker with detected objects
    tracker.update_objects(detected_objects, frame_count)

    # Mapping from detected object's coordinates back to their IDs for display
    obj_mapping = {v: k for k, v in tracker.get_objects().items()}

    # Annotation loop using original detected coordinates
    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                label = f"{names[int(cls)]}"
                converted_x, converted_y = mapper.convert(x1, y1)  # only for mapping
                obj_id = obj_mapping.get((label, (converted_x, converted_y)))

                if obj_id is not None:
                    # Place text using the original detected top-left corner coordinates
                    cv2.putText(frame, f"{label} ID: {obj_id} ({converted_x:.1f}, {converted_y:.1f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if frame_count % 15 == 0:  # Print every 15 frames
        print(f"Frame {frame_count}:")
        for obj_id, (obj_label, (x, y)) in tracker.get_objects().items():
            print(f"Object ID: {obj_id}, Label: {obj_label}, Coordinates: ({x:.1f}, {y:.1f})")

    return frame


def resizeFrameToMatch(frame1, frame2):
    """ Resize two frames to match their dimensions for stitching """
    height1, width1 = frame1.shape[:2]
    height2, width2 = frame2.shape[:2]
    # Determine scaling factor and resize accordingly
    if height1 < height2:
        scaling_factor = height2 / float(height1)
        frame1 = cv2.resize(frame1, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    elif height1 > height2:
        scaling_factor = height1 / float(height2)
        frame2 = cv2.resize(frame2, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame1, frame2

def stitchFrames(frame1, frame2):
    """ Stitch two frames horizontally after resizing to match dimensions """
    frame1, frame2 = resizeFrameToMatch(frame1, frame2)
    return np.hstack((frame1, frame2))

def resize_image(image, new_shape):
    """ Resize an image to a specified shape """
    return cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)


# # Comment/uncomment this for RTSP runs/testing
# rtsp_url1 = "rtsp://141.165.40.33/stream1"
# rtsp_url2 = "rtsp://141.165.40.34/stream1"
# cap1 = cv2.VideoCapture(rtsp_url1)
# cap2 = cv2.VideoCapture(rtsp_url2)
# frame_count = 0
# tracker = SimpleTracker()
# points1 = [(1265, 1047), (1579, 333), (865, 91), (413, 513)]
# points2 = [(1290, 982), (1526, 354), (979, 297), (596, 712)]
# matrix1, new_width1, new_height1 = computeTransformationMatrix(points1, rotate=True)
# matrix2, new_width2, new_height2 = computeTransformationMatrix(points2, rotate=False)
# while True:
#     ret1, frame1 = cap1.read()
#     ret2, frame2 = cap2.read()
#
#     if not ret1 or not ret2:
#         print("Failed to grab frames.")
#         break
#
#     transformed_frame1 = createTopDownView(frame1, matrix1, new_width1, new_height1, rotate=True)
#     transformed_frame2 = createTopDownView(frame2, matrix2, new_width2, new_height2)
#     stitched_frame = stitchFrames(transformed_frame2, transformed_frame1)
#
#     frame = detect(stitched_frame, frame_count, tracker)
#
#     cv2.imshow('Digital Twin - Object Detection', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# Comment/uncomment this for video runs/testing
cap = cv2.VideoCapture('New Videos/run3.mp4')
tracker = SimpleTracker()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = detect(frame, frame_count, tracker)  # Apply object detection to each frame
    cv2.imshow('Digital Twin - Object Detection', frame)  # Display the frame
    if cv2.waitKey(1) == ord('q'):  # Exit loop if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
