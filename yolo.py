import sys

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
import itertools
import time
import torchvision.transforms

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
        self.img_size = opt.img_size
        print('Yolo Initialized')

    def display(self, tensor):
        plt.figure()
        toPIL = torchvision.transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = toPIL(image)
        plt.imshow(image)
        plt.show()

    def get_obj(self, input_imgs) -> dict:
        # self.display(input_imgs)
        img, _ = pad_to_square(input_imgs, 0)
        img = resize(img, self.img_size)
        if len(list(img.shape)) < 4:
            img = img.unsqueeze(0)
        # self.display(img[0, 0])

        # Get detections
        with torch.no_grad():
            detections = self.model(img)
            detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)

        imgs = {('', ''): input_imgs}
        # Iterate through images and save plot of detections
        for _, detections in enumerate(detections):
            # Draw bounding boxes and labels of detections
            if detections is not None:
                # Rescale boxes to original image
                img = torch.zeros((1, 210, 160), device="cuda")
                detections = rescale_boxes(detections, self.opt.img_size, img.shape[1:])
                unique_labels = detections[:, -1].cpu().unique()
                # unique_labels = list(lambda x: self.classes[int(x)], unique_labels)
                unique_labels = [self.classes[int(x)] for x in unique_labels]
                unique_labels.sort()
                combos = []
                for i in unique_labels:
                    for j in unique_labels:
                        if j is not i:
                            combos.append((i, j))
                for combo in combos:
                    for i, (x1, y1, x2, y2, _, _, cls_pred) in enumerate(detections):
                        cls_pred = self.classes[int(cls_pred)]
                        for j, (jx1, jy1, jx2, jy2, _, _, jcls_pred) in enumerate(detections):
                            jcls_pred = self.classes[int(jcls_pred)]
                            if i is not j and cls_pred in combo and jcls_pred in combo:
                                img[:, int(x1):int(x2), int(y1):int(y2)] = 1
                                img[:, int(jx1):int(jx2), int(jy1):int(jy2)] = 1
                                imgs.update({combo: img})

        return imgs

    def pad_to_square(img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

        return img, pad

    def resize(image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image