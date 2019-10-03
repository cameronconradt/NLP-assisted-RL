import sys

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
import itertools
import time

import torch

class Yolo:
    def __init__(self, opt):
        self.opt = opt
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = Darknet(self.opt.model_def, img_size=opt.img_size).to(device)
        if self.opt.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.opt.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.opt.weights_path))

        self.model.eval()  # Set in evaluation mode

        self.classes = load_classes(self.opt.class_path)  # Extracts class labels from file

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def get_obj(self, input_imgs):

        # Get detections
        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)

        imgs = {}
        # Iterate through images and save plot of detections
        for _, detections in enumerate(detections):
            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                img = torch.from_numpy(np.zeros((3, 160, 210)))
                detections = rescale_boxes(detections, self.opt.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                unique_labels = map(lambda x: self.classes[int(x)], unique_labels)
                combos = [zip(x, unique_labels) for x in itertools.permutations(unique_labels, len(unique_labels))]
                for combo in combos:
                    for i, (x1, y1, x2, y2, _, _, cls_pred) in enumerate(detections):
                        for j, (jx1, jy1, jx2, jy2, _, _, jcls_pred) in enumerate(detections):
                            if i is not j and cls_pred in combo and jcls_pred in combo:
                                img[:, x1:x2, y1:y2] = 1
                                img[:, jx1:jx2, jy1:jy2] = 1
                                imgs.update({combo: img})

        return imgs
