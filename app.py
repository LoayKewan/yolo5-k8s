import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
import json
import uuid
import requests

images_bucket = os.environ['BUCKET_NAME']
queue_name = os.environ['SQS_QUEUE_NAME']
region_of_sqs = os.environ['region_of_sqs']

sqs_client = boto3.client('sqs', region_name=region_of_sqs)


## new fix

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']
current_time = time.time()

def consume():
    while True:
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        logger.info(f'response is {response}')
        logger.info('loay **********---*******')

        if 'Messages' in response:
            message = response['Messages'][0]['Body']

            message = message.replace("'", "\"")
            my_string_dic = json.loads(message)

            photo_path = my_string_dic.get('photo_path')

            receipt_handle = response['Messages'][0]['ReceiptHandle']

            # Use the ReceiptHandle as a prediction UUID
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'prediction: {prediction_id}. start processing')

            # Receives a URL parameter representing the image to download from S3
            img_name = photo_path
            chat_id = my_string_dic.get('chat_id')

            logger.info(f'chat id is : {chat_id}')

            logger.info(f'photo_path is: {photo_path}')

            photo_S3_name = photo_path.split("/")
            file_path_pic_download = os.getcwd() + "/" + str(photo_S3_name[1])
            logger.info(file_path_pic_download)
            client = boto3.client('s3')
            client.download_file(images_bucket, str(photo_S3_name[1]), file_path_pic_download)

            # TODO download img_name from S3, store the local image path in the original_img_path variable.
            #  The bucket name is provided as an env var BUCKET_NAME.

            # original_img_path = file_path_pic_download
            original_img_path = file_path_pic_download
            logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')
            # This is the path for the predicted image with labels
            # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
            path = Path(f'static/data/{prediction_id}/{str(photo_S3_name[1])}')
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                pass
            predicted_img_path = Path(f'static/data/{prediction_id}/{str(photo_S3_name[1])}')
            path_str = str(predicted_img_path)
            json_str = json.dumps({"path": path_str})
            json_data = json.loads(json_str)
            unique_filename = str(uuid.uuid4()) + '.jpeg'
            client.upload_file(json_data["path"], images_bucket, unique_filename)
            # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).

            # Parse prediction labels and create a summary
            path = Path(f'static/data/{prediction_id}/labels/{photo_S3_name[1].split(".")[0]}.txt')
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                pass
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{photo_S3_name[1].split(".")[0]}.txt')
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': float(l[1]),
                        'cy': float(l[2]),
                        'width': float(l[3]),
                        'height': float(l[4]),
                    } for l in labels]

                    logger.info(f'prediction: {prediction_id}/{photo_S3_name[1]}. prediction summary:\n\n{labels}')

                    prediction_summary = {
                        'prediction_id': prediction_id,
                        'original_img_path': photo_S3_name[1],
                        'predicted_img_path': json_data["path"],
                        'labels': labels,
                        'time': current_time
                    }

            logger.info(f'prediction_summary is {prediction_summary}')

            # TODO store the prediction_summary in a DynamoDB table

            # Create a DynamoDB client
            dynamodb = boto3.client('dynamodb', region_name='eu-west-3')

            # Define the table name
            table_name = 'loay-PolybotService-DynamoDB-tf'

            # Define the prediction_summary data
            prediction_id = prediction_summary['prediction_id']
            original_img_path = prediction_summary['original_img_path']
            predicted_img_path = prediction_summary['predicted_img_path']
            labels = prediction_summary['labels']
            time = prediction_summary['time']

            # Insert the prediction_summary data into DynamoDB table
            try:

                labels_list = [{'M': {
                    'class': {'S': label['class']},
                    'cx': {'N': str(label['cx'])},
                    'cy': {'N': str(label['cy'])},
                    'width': {'N': str(label['width'])},
                    'height': {'N': str(label['height'])}
                }} for label in labels]

                response = dynamodb.put_item(
                    TableName=table_name, Item={
                        'chat_id': {'N': str(chat_id)},
                        'prediction_id': {'S': prediction_id},
                        'original_img_path': {'S': original_img_path},
                        'predicted_img_path': {'S': predicted_img_path},
                        'labels': {'L': labels_list},
                        'time': {'N': str(time)}
                })

                logger.info(f'data insertred to Dynamodb successfully')
            except botocore.exceptions.ClientError as e:
                # Handle errors or exceptions
                print(f"Error: {e}")

            # TODO perform a GET request to Polybot to `/results` endpoint

            # Define the base URL of the Polybot server

            polybot_base_url = "http://polybottest-service:8443"

            # Combine the base URL and the endpoint to form the complete URL
            url = polybot_base_url + f"/results?predictionId={prediction_id}"


            # Make the GET request to the endpoin

            headers = {'Content-Type': 'application/json'}  # Set the Content-Type header to JSON

            prediction_summary = {
                'prediction_id': prediction_id,
                'original_img_path': photo_S3_name[1],
                'predicted_img_path': json_data["path"],
                'labels': labels,
                'time': current_time}
            data = {'prediction_id': prediction_summary['prediction_id'],
                    'original_img_path': prediction_summary['original_img_path'],
                    'predicted_img_path': prediction_summary['predicted_img_path'],
                    'labels': prediction_summary['labels'], 'time': prediction_summary['time']}

            try:

                response = requests.post(url, headers=headers, json=data)

                response.raise_for_status()

                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        print(response_json)
                        # Process the JSON response data as needed
                    except json.JSONDecodeError as e:
                        print("Error decoding JSON response:", e)

                else:
                    print(f"Failed request. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print("Request error:", e)

            # Delete the message from the queue as the job is considered as DONE
            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)


if __name__ == "__main__":
    consume()


