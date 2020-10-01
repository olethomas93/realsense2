import cv2
import numpy as np


def detect_people(frame, net):

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat", "bottle"
                  "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike",  "pottedplant", "sheep",
                  "sofa", "train", "tvmonitor"])

    (h, w) = frame.shape[:2]

    frame_resized = cv2.resize(frame, (300, 300))

    blob = cv2.dnn.blobFromImage(
        frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    centroids = []
    confidences = []

    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    # loop over the detections
    for i in range(detections.shape[2]):

        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > float(0.75):

            classID = int(detections[0, 0, i, 1])  # class label

            # if the predicted class label is in the set of classes
            # we want to ignore then skip the detection
            if CLASSES[classID] in IGNORE:
                continue

             # Object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0
            widthFactor = frame.shape[1]/300.0
            # Scale object detection to frame
            startX = int(widthFactor * xLeftBottom)
            startY = int(heightFactor * yLeftBottom)
            endX = int(widthFactor * xRightTop)
            endY = int(heightFactor * yRightTop)

            # compute the (x, y)-coordinates of the bounding box for
            # the object
            # box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # (startX, startY, endX, endY) = box.astype("int")
            # w, h = (endX-startX, endY-startY)

            predbox = (abs(startX), abs(startY), abs(endX), abs(endY))
            boxes.append(predbox)

    return boxes
