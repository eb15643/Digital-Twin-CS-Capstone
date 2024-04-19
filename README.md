# Digital Twin - Capstone Project - Spring 2024

A digital twin is a virtual model designed to accurately mirror a physical object, system, or process. It integrates real-time data to simulate, predict, and optimize performance, enhancing decision-making and operational efficiency. This technology is increasingly utilized across industries to anticipate problems and test solutions in a cost-effective virtual environment

## Expectation
For our capstone project, we aimed to employ an object detection algorithm, such as YOLO, to analyze video camera feeds and accurately track robot movements across a grid. Our objective was to detect when a robot picks up or sets down a rack, effectively replicating all robot movement within the actual lab environment. The data extracted from the Python scripts and camera feeds—including X and Y coordinates and object types like "robot" and "rack"—would then be uploaded to the DynamoDB database table. Utilizing the DynamoDB blueprints plugin that we acquired, we planned to query this database table to retrieve the necessary information. This data would then allow us to simulate the robots' movements in the lab using a virtual environment created in Unreal Engine 5 and various 3D models developed in Blender.

## Gulfstream Sponsorship

From 2008, when the G650 was introduced, Gulfstream has been using 3D CAD data to certify and produce all in-production aircraft. With this massive amount of 3D data, the XR team has been able to create several XR simulations.

As game engines become more commonplace in the enterprise space, European
automotive companies have paved the way for the generation of ‘digital twins’ of their products. Gulfstream’s XR team is interested in pursuing a true digital twin of a facility within one of the popular game engines (Unreal 5).

The goal of this project is to advance research begun by the Capstone group in Fall 2023. Security Cameras within the engineering building focused on the APRN: Rows lab will be fed into a system running Computer Vision/Object Detection to identify robots, racks, and robots carrying racks, and locate them within the building. One such application of this capability would be to alert a user that a robot is getting too close to an object it shouldn't.


## Fall '23 Details

Object Detection Algorithms: Implementing the YOLO (You Only Look Once) algorithm, we seamlessly integrated it with the camera footage. This enabled real-time tracking of moving objects, with data promptly relayed to a MongoDB database. The model, trained using CVAT.ai and over 500 annotated images, achieved an impressive average delay of 52ms.

Database Management: Connecting Unreal Engine with MongoDB using C++, we established a streamlined process for data exchange. Pixel coordinates from the object detector were translated into Unreal coordinates using a defined equation: Unreal Coordinate = Scale * Pixel Coordinate - Offset.

3D Modeling Technology: For a more refined model, especially tailored to objects resembling our robots, we employed photogrammetry. Reality Capture served as the primary tool, supplemented by LumaAI & Blender for final touches and improved results.

In summary, our project seamlessly integrated multi-perspective views, advanced object detection algorithms, efficient database management, and cutting-edge 3D modeling techniques. The digital twin created is not only adaptable but also serves as a testament to the convergence of technologies, offering a comprehensive solution for real-time monitoring and analysis within our university lab.

[Link](https://www.youtube.com/watch?v=KnP4f9_9hJw) for a video of their final product.

(Fall '23 Members: Jensen Bromm, Jerome Larson, Joselin Aguirre Trujillo, and Mateo Maldonado Rojas)

## Spring '24 Details

Due to the inability to load the Unreal Engine project started by the Fall group (primarily as a result of the custom MongoDB plugin that we were unable to replicate), our team made the decision to essentially start the project over from the beginning, with a key focus on documentation for next semesters group to be able to effectively pick up where we leave off.

**Object Detection:** The YOLO (You Only Look Once) algorithm was confirmed to be a good selection for this project. We started with the custom trained model the previous semester completed, but found that its accuracy needed improvement in order to detect certain objects.  Two version of YOLO were tested.  First was YOLOv8, which was the same model, but a new set of annotated images were used that more accurately represented the top-down view of the objects that the algorithm would be run against. The results are as follows:

Training Losses:\
Box Loss: Decreased by 47.81%\
Classification Loss: Decreased by 45.75%\
Direction-Focused Loss (DFL): Decreased by 17.77%\
Metrics (B):\
Precision: Increased by 4.38% \
Recall: Increased by 5.68%\
mAP50: Increased by 1.94%\
mAP50-95: Increased by 19.25%

Validation Losses:\
Box Loss: Decreased by 41.48%\
Classification Loss: Decreased by 45.29%\
Direction-Focused Loss (DFL): Decreased by 11.08%\
Learning Rates (lr/pg0, lr/pg1, lr/pg2): Increased dramatically by 3843.50%\

The biggest concern of v8 is high inference times, and as the goal is real-time detection, we tried YOLOv5 as well.  While not as accurate, the inference times allowed for real-time detection.

**Database Management:** It was determined that deploying a successful Unreal Engine plugin was a whole project in its self. A decision was made to purchase a pre-built blueprint plugin for use with AWS DynamoDB as the new database management option. Teams continuing this project in future semesters will also be required to purchase this [plugin](https://www.unrealengine.com/marketplace/en-US/product/awscore-dynamodb?sessionInvalidated=true) (reimbursement through Capstone can be sought) in order to successfully run this project.

**3D Modeling:** Blender was the choice in software for our group as well.  A complete 3D replica of the APRN: Rows lab was designed to attempt a realistic Digital Twin environment.

##  Spring '24 Members

Hayes Young (Team Lead), Evan Brantley, Shayna Gulley, and Changgyun Han


