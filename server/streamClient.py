import socket
import sys
import cv2
import pickle
import numpy as np
import struct
import zlib
import time
import ssl
import face_recognition
import threading


class streamClient(threading.Thread):

    def __init__(self, host, port):

        threading.Thread.__init__(self)
        self.ip = host
        self.port = port

    def run(self):

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        print('Socket created')
        # client.sendto(bytesToSend, serverAddressPort)

        client.connect((self.ip, self.port))
        print('Socket bind complete')

        print('Socket now listening')

        data = b""
        payload_size = struct.calcsize("Q")
        print("payload_size: {}".format(payload_size))

        i = 0
        while True:

            start = time.time()
            while len(data) < payload_size:
                packet = client.recv(4*1024)  # 4K
                if not packet:
                    break
                data += packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += client.recv(4*1024)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data)
            cv2.imshow('ImageWindow', frame)
            cv2.waitKey(1)

            stop = time.time()
            if (i % 10) == 0:
                i += 1
                # print("FPS: " + str(1/(stop - start)))
                # Hit 'q' on the keyboard to quit!

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break
                self.stopped = True

        client.close()

    def predict(img, knn_clf=None, model_path=None, distance_threshold=0.6):
        """
        Recognizes faces in given image using a trained KNN classifier
        :param X_img_path: path to image to be recognized
        :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
            of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """

        if knn_clf is None and model_path is None:
            raise Exception(
                "Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        X_face_locations = face_recognition.face_locations(img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(
            img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <=
                       distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def findFace(frame):

    face_locations = face_recognition.face_locations(frame)
    # face_encodings = face_recognition.face_encodings(frame, face_locations)

    return face_locations
