# <div align="center">Faster Pose Estimation</div>

<div align="center">
<p>WHOLE BODY, BODY AND HAND POSE ESTIMATION with YOLOv5 and MMDetection</p>
<p>
<img src="images/girl.gif" width="270"/> <img src="images/gin.gif" width="270"/> 
</p>
</div>

## Requirements

* mmcv >= 1.4.0
* mmpose >= 0.26.0

Install MMCV and MMPose according to  [documentation](https://mmpose.readthedocs.io/en/latest/install.html)

Other requirements can be installed with `pip install -r requirements.txt`.

Clone the repository recursively:

```bash
$ git clone --recursive git@github.com:1chimaruGin/Pose-Estimation.git
```

Then download a YOLO model's weights,

Hand pose estimation - [hand](https://drive.google.com/file/d/1a37j_8OJ8iZJQJ9qdS5kNZOFsDKvDtM4/view?usp=sharing)

All body and body estimation - [body](https://drive.google.com/file/d/1tKSvokFw-iadEJL81c9HAGKttNosdYYi/view?usp=sharing)

Place them in `weights/`.

## Pose Esitmation

Change the config acording to your needs.

```
DETMODEL:
  CHECKPOINT            : weights/yolov5s6.pt
  HALF                  : False
  CLASSES               : 0
  IMGSZ                 : 1280
  CONF_THRESH           : 0.5
  IOU_THRESH            : 0.5

POSEMODEL:
  CONFIG                : configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py
  CHECKPOINT            : https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth
```

```bash
$ python pose.py --source /path/to/video.mp4
```

## References

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmpose

## Citations

``` 
{
  @misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
