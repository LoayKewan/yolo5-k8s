FROM ultralytics/yolov5:latest-cpu
WORKDIR /usr/src/app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

COPY . .

ENV BUCKET_NAME=loayk-bucket-tf
ENV SQS_QUEUE_NAME=loay-PolybotServiceQueue-tf

CMD ["python3", "app.py"]
