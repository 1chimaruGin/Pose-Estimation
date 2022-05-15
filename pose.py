import cv2
import sys
import torch
import argparse
import warnings
import numpy as np
from omegaconf import OmegaConf
from typing import List, Tuple

from helpers.boxes import non_max_suppression, scale_coords
from helpers.helper import ReadVideo, check_img_size

from mmpose.datasets import DatasetInfo
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    vis_pose_result,
)

sys.path.insert(0, "yolov5")
from yolov5.models.common import DetectMultiBackend


class Detector:
    def __init__(
        self, config: str,
    ) -> None:
        conf = OmegaConf.load(config)
        det_conf = conf['DETMODEL']
        pose_conf = conf['POSEMODEL']
        model = det_conf.CHECKPOINT
        self.conf_thresh = det_conf.CONF_THRESH
        self.iou_thresh = det_conf.IOU_THRESH
        self.classes = conf.CLASSES if det_conf.CLASSES else [0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = DetectMultiBackend(model, device=self.device, dnn=False)

        pose_model = init_pose_model(pose_conf['CONFIG'], pose_conf['CHECKPOINT'], device=self.device)
        self.dataset = pose_model.cfg.data["test"]["type"]
        # get datasetinfo
        dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
        if dataset_info is None:
            warnings.warn(
                "Please set `dataset_info` in the config."
                "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
                DeprecationWarning,
            )
        else:
            dataset_info = DatasetInfo(dataset_info)

        self.dataset_info = dataset_info
        self.pose_model = pose_model

        half = det_conf.HALF

        stride, _, pt, jit, onnx, engine = (
            model.stride,
            model.names,
            model.pt,
            model.jit,
            model.onnx,
            model.engine,
        )
        self.model = model
        self.imgsz = check_img_size(det_conf.IMGSZ, s=stride)  # check image size
        self.stride = stride
        self.pt = pt

        half &= (
            pt or jit or onnx or engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        self.half = half
        if pt or jit:
            model.model.half() if self.half else model.model.float()

    def prepare(self, img: np.ndarray) -> torch.Tensor:
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def post_process(
        self, pred: torch.Tensor, im: torch.Tensor, im0: torch.Tensor
    ) -> torch.Tensor:
        pred = non_max_suppression(
            pred, self.conf_thresh, self.iou_thresh, classes=self.classes
        )
        output = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, cls, _ in reversed(det.to("cpu").numpy()):
                    output.append({"bbox": [*xyxy, cls]})
        return output

    @torch.no_grad()
    def predict(
        self, img: np.ndarray, im0: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        im = self.prepare(img)
        pred = self.model(im)
        output = self.post_process(pred, im, im0)
        self.pose_inference(im0, output, show=True)

    def pose_inference(self, im0, out, show: bool = False):
        pose_model = self.pose_model
        dataset = self.dataset
        dataset_info = self.dataset_info
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            im0,
            out,
            bbox_thr=0.5,
            format="xyxy",
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        vis_frame = vis_pose_result(
            pose_model,
            im0,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=0.35,
            radius=4,
            thickness=1,
            show=False,
        )

        if show:
            cv2.imshow("Frame", vis_frame)

        if show and cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()


def pose_estimate(
    source: str, cfg: str) -> None:
    detector = Detector(
        config=cfg
    )
    stride = detector.stride

    dataset = ReadVideo(source, stride=stride)
    for i, (img, im0) in enumerate(dataset):
        detector.predict(img, im0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Source Video")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        type=str,
        help="Detector Config",
    )

    args = parser.parse_args()

    if not args.source:
        print("Please specify a source video (e.g --source videos/test.mp4)")
        exit(1)

    pose_estimate(
        source=args.source,
        cfg=args.config,
    )
