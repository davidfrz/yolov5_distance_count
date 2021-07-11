# yolov5-双目摄像头
通过yolov5实现目标检测+双目摄像头实现距离测量<br>
<div align="left"> <img src="./pic/1.png" height="320"> </div>
示例视频 ===>[https://www.bilibili.com/video/BV1QK411w71d]<br>

## yolov5
本项目基于yolov5 ===>[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)<br>
我把v5s那个权重放在里面了 你可以加任意的权重 自己训练的权重也行

## 版本号
版本号选择的是3.1  当然  这个项目和什么版本是无关的 你可以用任意版本（至少到目前是这样子的）

## 如何使用
在原始的 "yolov5" 中添加了3个文件 --->
camera_config.py
dis_count.py
video_remain.py

|文件|说明|
|----|----|
|camera_config.py|双目摄像头参数|
|dis_count.py|深度图+距离矩阵|
|video_remain.py|主函数|

## RESULT
|DEVICEE|FPS|
|----|----|
|1650|20|
|TX2|12|
|NX|15|

按理来说啊~接上你的双目摄像头直接运行vedio_remain.py这个文件就能实现了

如果这个项目对你有帮助 请帮我点个星星吧  谢谢啦~

那个如果有啥想联系的 可以直接在github里面留言问我

如果你想加我联系 QQ： 772452210
