import argparse
import time
from pathlib import Path
import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from dis_count import *
from utils.general import xyxy2xywh, xywh2xyxy
from utils.torch_utils import torch_distributed_zero_first
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from dis_count import *
from utils.datasets import *


# 设置model
device = torch.device('cuda:0')
half = device.type != True  # half precision only supported on CUDA

model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model
imgsz = check_img_size(640, s=model.stride.max())  # check img_size

if half:
    model.half()  # to FP16

view_img = True
cudnn.benchmark = True  # set True to speed up constant image size inference

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

img01 = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img01.half() if half else img01) if device.type != 'cpu' else None  # run once

# cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(2)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):
    ret, frame = camera.read()

    #  通过切片操作将原始图像frame分割成了左侧图像和右侧图像
    left_frame = frame[0:480, 0:640]
    right_frame = frame[0:480, 640:1280]

    imgs = [None] * 1
    imgs2 = [None] * 1

    camera.grab()

    imgs[0] = left_frame
    imgs2[0] = right_frame



    # _, imgs[0] = cap1.retrieve()
    # _, imgs2[0] = cap2.retrieve()

    img = [letterbox(x, new_shape=640, auto=True)[0] for x in imgs]
    # imgb = [letterbox(x1, new_shape=640, auto=True)[0] for x1 in imgs2]

    # Stack
    img = np.stack(img, 0)
    # imgb = np.stack(imgb, 0)

    # Convert

    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)

    # imgb = imgb[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    # imgb = np.ascontiguousarray(imgb)


    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0   # torch.Size([1, 3, 480, 640])

    # imgg = torch.from_numpy(imgb).to(device)
    # imgg = imgg.half() if half else imgg.float()  # uint8 to fp16/32
    # imgg /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # if imgg.ndimension() == 3:
    #     imgg = imgg.unsqueeze(0)

    pred = model(img, augment=False)[0]
    # predd = model(imgg, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    # predd = non_max_suppression(predd, 0.25, 0.45, classes=None, agnostic=False)

    dislist,disp = dis_co(imgs[0], imgs2[0])
    # dislist=torch.from_numpy(dislist)
    def ved(pred):
        # t0 = time.time()
        for i, det in enumerate(pred):  # detections per image

            dis_box = dict()
            if True:  # batch_size >= 1
                 s, im0 =  '%g: ' % i, imgs[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                # print(det)
                dddd = 0
                d1=0
                for *xyxy, conf, cls in reversed(det):
                    x = ((xyxy[2]-xyxy[0])/2)+xyxy[0]
                    y = ((xyxy[3]-xyxy[1])/2)+xyxy[1]
                    if(x<192):
                        pos='left'
                    elif((x<448) and (x>192)):
                        pos='mid'
                    else:
                        pos='right'
                    x=int(x.cpu())
                    y=int(y.cpu())
                    # print(xyxy)
                    dddd=(dislist[y][x]/5)[-1]
                    label = '%s %.2f %.2f %s' % (names[int(cls)], conf, dddd,pos)
                    msg={pos:dddd}
                    dis_box.update(msg)
                    # print(label)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        # print('Done. (%.3fs)' % (time.time() - t0))
        pos_msg=''
        dis_msg=0.
        for key, value in dis_box.items():
            if (value == min(dis_box.values())):
                print(key,value)
                # pos_msg=key
                # dis_msg=value
        return im0
    def vedd(pred):
        for i, det in enumerate(pred):  # detections per image
            if True:  # batch_size >= 1
                 s, im0 = '%g: ' % i, imgs2[i].copy()

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        return im0

    # 计算两针深度图  左帧img 右帧imgg
    v1 = ved(pred)
    # v2 = vedd(predd)
    dis_box = dict()
    dislist=np.ndarray(0)
    cv2.imshow('0', v1)
    # cv2.imshow('1', v2)
    cv2.imshow('SGNM_disparity', (disp - 0) / 32)

    c = cv2.waitKey(1) & 0xff
    if c == 27:
        camera.release()
        break