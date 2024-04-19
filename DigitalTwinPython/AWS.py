import boto3
from decimal import Decimal

def convert_floats_to_decimals(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = convert_floats_to_decimals(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = convert_floats_to_decimals(item)
    return obj

def dynamodb_writer(queue):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('DigitalTwin')

    while True:
        items = queue.get()
        if items == 'SHUTDOWN':
            break

        for detection in items:
            # Convert floats to Decimals for DynamoDB compatibility
            detection = convert_floats_to_decimals(detection)

            # Construct the DynamoDB item with bounding box coordinates
            dynamo_item = {
                'ObjectID': str(detection['objectID']),  # Ensure ObjectID is a string
                'Timestamp': detection['timestamp'],
                'ObjectType': detection['object_type'],
                'X': detection['X'],
                'Y': detection['Y'],
                'CameraNumber': detection['camera_num']  # Include camera number in the item
            }

            try:
                table.put_item(Item=dynamo_item)
                print("Item uploaded successfully.")
            except Exception as e:
                print(f"Error uploading item: {e}")
