# Understanding Python Files

**AWS.py:** Simple DynamoDB writer for use within ObjectDetection.py. In order to use this, you must have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed and configured.

**CameraTransformation.py:** Contains the functions that get applied to the RTSP feed to transform the video and create a top-down view. 

**ObjectDetection.py:** This is the main script that uses YOLOv5 to detect objects in either a video file or the RTSP feeds for live real-time data. Coordinates are converted from the bounding box information of the YOLO detection to a simpler 9x4 grid for Unreal coordinate usage. A very simple rudimentary tracker was made to give consistent IDs to objects across frames.

**StitchTest.py:** Used to test various points for the transformation matrix and show what the final stitched RTSP feeds will look like.


