import numpy as np
import torch
from configparser import ConfigParser


class YOLO:
    def __init__(self):
        with open("config.cfg", "r") as file:
            config = ConfigParser()
            config.read_file(file)

        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="models/detector.pt", force_reload=True)
        self.model.conf = config["Parameters"].getfloat("conf_threshold")
        self.model.iou = config["Parameters"].getfloat("iou_threshold")
        self.size = config["Parameters"].getint("size")

    def forward(self, image):
        return self.model(image[:, :, ::-1].astype(np.uint8), size=self.size).pandas().xyxy[0].to_numpy()

    @staticmethod
    def get_centroid(detection):
        x = int(detection[0] + (detection[2] - detection[0]) / 2)
        y = int(detection[1] + (detection[3] - detection[1]) / 2)

        return x, y

    @staticmethod
    def get_crop(image, detection):
        return image[int(detection[1]):int(detection[3]), int(detection[0]):int(detection[2])]
