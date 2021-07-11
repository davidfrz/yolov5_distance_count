# yolov5-Binocular camera-distance count-ranging
中文文档 --->[https://github.com/davidfrz/yolov5_distance_count/blob/master/README_CN.md](https://github.com/davidfrz/yolov5_distance_count/blob/master/README_CN.md)<br>
<div align="left"> <img src="./pic/1.png" height="320"> </div>
This project can get object recognition and distance display of the measured object through YOLOV5 target detection box with binocular camera.<br>
Sample Vedio ===>[https://www.bilibili.com/video/BV1QK411w71d]<br>

## yolov5
All project is based on yolov5 ===>[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)<br>
And I put weight(yolov5s.pt)  inside.

## version
I chose the v3.1 version, but this project is just a matter of adding a few .py files, whatever version will do (at least for now).

## HOW TO USE
I add three files in origin "yolov5" --->
camera_config.py
dis_count.py
video_remain.py

|Files|instructions|
|----|----|
|camera_config.py|Binocular camera parameters|
|dis_count.py|Depth map, distance matrix|
|video_remain.py|main|

## RESULT
|DEVICEE|FPS|
|----|----|
|1650|20|
|TX2|12|
|NX|15|

JUST RUN video_remain.py and you can get what you want.

if this project can help you, give me a star,plzzzzzz! 
