import socket
import threading
import cv2
import numpy as np
import sys
import pickle
import sys
import struct
import logging
import ssl


class ClientThread(threading.Thread):

    def __init__(self, clientAddress, clientsocket, stream):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        self.clientAddress = clientAddress
        self.stream = stream
        self.lock = threading.Lock()
        (self.grabbed, self.frame) = self.stream.read()
        print("New connection added: ", clientAddress)
        self.stopped = False

    def run(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        print("Connection from : ", self.clientAddress)

        while not self.stopped:

            print('waiting for lock')
            self.lock.acquire()

            try:
                (self.grabbed, self.frame) = self.stream.read()

                a = pickle.dumps(self.frame, 0)
                message = struct.pack("Q", len(a))+a
            except:
                print('error')
            finally:
                logging.debug('Released a lock')
                self.csocket.sendall(message)
                self.lock.release()

        cam.release()
        print("Client at ", self.clientAddress, " disconnected...")


class ServerThread(threading.Thread):

    def __init__(self, host, port):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port

    def run(self):

        ThreadCount = 0

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # server = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        lock = threading.Lock()
        stream = cv2.VideoCapture(0)

        print("Server started")
        print("Waiting for client request..")
        while True:
            server.listen(5)
            clientsocket, clientAddress = server.accept()
            newthread = ClientThread(clientAddress, clientsocket, stream)
            ThreadCount += 1
            newthread.start()
        server.close()
