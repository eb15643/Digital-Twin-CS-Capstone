import cv2
import numpy as np

def createTopDownView(frame, matrix, new_width, new_height, target_width=None, target_height=None, rotate=False):
    # Apply the perspective warp to create the top-down view
    top_down_view = cv2.warpPerspective(frame, matrix, (new_width, new_height))

    # Optionally lower the resolution of the top-down view
    if target_width and target_height:
        top_down_view = cv2.resize(top_down_view, (target_width, target_height))

    if rotate:
        # Flip vertically and then rotate counter-clockwise
        top_down_view = cv2.flip(top_down_view, 0)
        top_down_view = cv2.rotate(top_down_view, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Just flip horizontally (updated as per your requirement)
        top_down_view = cv2.flip(top_down_view, 0)

    return top_down_view

def computeTransformationMatrix(points, rotate=False):
    new_width, new_height = (1080, 960) if rotate else (960, 1080)
    new_points = np.array([
        [0, 0],
        [new_width - 1, 0],
        [new_width - 1, new_height - 1],
        [0, new_height - 1]], dtype=np.float32)
    original_points = np.array(points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(original_points, new_points)
    return matrix, new_width, new_height
