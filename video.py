import argparse
import time
from pathlib import Path

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
import camera_configs
fpss=0.0
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = True

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # print(half)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap,imgg,im0ss,aaaa,bbbb in dataset:
        # print(img.shape)  (1, 3, 480, 640)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0   # torch.Size([1, 3, 480, 640])

        imgg = torch.from_numpy(imgg).to(device)
        imgg = imgg.half() if half else imgg.float()  # uint8 to fp16/32
        imgg /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if imgg.ndimension() == 3:
            imgg = imgg.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        predd = model(imgg, augment=opt.augment)[0]
        # print('********************')
        # print(opt.augment)
        # print('********************')
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        predd = non_max_suppression(predd, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # print(pred)
        def ved(pred):
            for i, det in enumerate(pred):  # detections per image
                dis_box=[]
                if webcam:  # batch_size >= 1
                    p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()

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
                    dddd=0.
                    for *xyxy, conf, cls in reversed(det):


                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        x=xywh[0]
                        y=xywh[1]
                        # dddd=dis_co(aaaa,bbbb,x,y)

                        label = '%s %.2f %.2f' % (names[int(cls)], conf,dddd)
                        # print(label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            return im0
        def vedd(pred):
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = Path(path[i]), '%g: ' % i, im0ss[i].copy()
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '_%g' % dataset.frame if dataset.mode == 'video' else '')
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
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image

                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            return im0
        #计算两针深度图  左帧img 右帧imgg
        v1=ved(pred)
        v2=vedd(predd)

        cv2.imshow('0',v1)
        cv2.imshow('1', v2)

        # Process detections
        # print(pred) # list

        # for i, det in enumerate(pred):  # detections per image
        #     if webcam:  # batch_size >= 1
        #         p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
        #     else:
        #         p, s, im0 = Path(path), '', im0s
        #
        #     save_path = str(save_dir / p.name)
        #     txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        #     s += '%gx%g ' % img.shape[2:]  # print string
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        #
        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
        #             s += '%g %ss, ' % (n, names[int(c)])  # add to string
        #
        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             if save_txt:  # Write to file
        #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                 line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
        #                 with open(txt_path + '.txt', 'a') as f:
        #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')
        #
        #             if save_img or view_img:  # Add bbox to image
        #                 label = '%s %.2f' % (names[int(cls)], conf)
        #                 plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        #
        #     # Print time (inference + NMS)
        #     print('%sDone. (%.3fs)' % (s, t2 - t1))
        #     global fpss
        #     # Stream results
        #
        #     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
        #     fpss=(fpss+(1./(t2-t1)))/2
        #     im0 = cv2.putText(im0, "fps= %.2f" % (fpss), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     cv2.imshow(str(p), im0)
        #     # 在这里让im0变成两个  这样的话重新展示会有两个框
        #     if cv2.waitKey(1) == ord('q'):  # q to quit
        #         raise StopIteration
        # for i, det in enumerate(predd):  # detections per image
        #     if webcam:  # batch_size >= 1
        #         p1, s1, im01 = Path(path[i]), '%g: ' % i, im0ss[i].copy()
        #     else:
        #         p1, s1, im01 = Path(path), '', im0ss
        #
        #     txt_path = str(save_dir / 'labels' / p1.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        #     s1 += '%gx%g ' % imgg.shape[2:]  # print string
        #     gn = torch.tensor(im01.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(imgg.shape[2:], det[:, :4], im01.shape).round()
        #
        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
        #             s1 += '%g %ss, ' % (n, names[int(c)])  # add to string
        #
        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             if save_txt:  # Write to file
        #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                 line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
        #                 with open(txt_path + '.txt', 'a') as f:
        #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')
        #
        #             if save_img or view_img:  # Add bbox to image
        #                 label = '%s %.2f' % (names[int(cls)], conf)
        #                 plot_one_box(xyxy, im01, label=label, color=colors[int(cls)], line_thickness=3)
        #
        #     # Print time (inference + NMS)
        #     print('%sDone. (%.3fs)' % (s1, t2 - t1))
        #
        #     # Stream results
        #
        #     cv2.namedWindow(str(p1), cv2.WINDOW_NORMAL)
        #     cv2.imshow(str(p1), im01)
        #     # 在这里让im0变成两个  这样的话重新展示会有两个框
        #     if cv2.waitKey(1) == ord('q'):  # q to quit
        #         raise StopIteration

        print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0,1', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
