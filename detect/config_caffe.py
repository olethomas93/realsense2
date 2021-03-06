# base path to YOLO directory
MODEL_PATH = "ssd_net"

LABEL = "person"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.5
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# minimum distance meter
MIN_DISTANCE = 1.0
