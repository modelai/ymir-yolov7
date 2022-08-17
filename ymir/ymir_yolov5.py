"""
utils for ymir and yolov5
"""

import glob
import os
import os.path as osp
import shutil
from enum import IntEnum
from typing import Any, List

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from nptyping import NDArray, Shape, UInt8
from packaging.version import Version
from ymir_exc import env
from ymir_exc import result_writer as rw
from ymir_exc.util import get_weight_files

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class YmirStage(IntEnum):
    PREPROCESS = 1  # convert dataset
    TASK = 2  # training/mining/infer
    POSTPROCESS = 3  # export model


BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def get_weight_file(cfg: edict) -> str:
    """
    cfg: from get_merged_config()
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path
    return the weight file path by priority
    """
    weight_files = get_weight_files(cfg, suffix=('.pt'))

    # choose weight file by priority, best.pt > xxx.pt
    for f in weight_files:
        if f.endswith('best.pt'):
            return f

    if len(weight_files) > 0:
        return max(weight_files, key=osp.getctime)

    return ""


class YmirYolov5(object):
    """
    used for mining and inference to init detector and predict.
    """

    def __init__(self, cfg: edict):
        self.cfg = cfg
        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            # multiple task, run mining first, infer later
            infer_task_idx = 1
            task_num = 2
        else:
            infer_task_idx = 0
            task_num = 1

        self.task_idx = infer_task_idx
        self.task_num = task_num

        device = select_device(cfg.param.get('gpu_id', 'cpu'))

        self.model = self.init_detector(device)
        self.device = device
        self.class_names = cfg.param.class_names
        self.stride = int(self.model.stride.max())
        self.conf_thres = float(cfg.param.conf_thres)
        self.iou_thres = float(cfg.param.iou_thres)

        img_size = int(cfg.param.img_size)
        # imgsz = (img_size, img_size)
        imgsz = check_img_size(img_size, s=self.stride)
        self.img_size = (imgsz, imgsz)

        self.half = device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        if self.half:
            # Run inference, warm up
            self.model(
                torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                    next(self.model.parameters())))

    def init_detector(self, device: torch.device) -> Any:
        weights = get_weight_file(self.cfg)

        if not weights:
            raise Exception('not weights file found!!')
        model = attempt_load(weights, map_location=device)
        return model

    def predict(self, img: CV_IMAGE) -> NDArray:
        """
        predict single image and return bbox information
        img: opencv BGR, uint8 format
        """
        # preprocess: padded resize
        img1 = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]

        # preprocess: convert data format
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = torch.from_numpy(img1).to(self.device)
        img1 = img1.half() if self.half else img1.float()
        img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
        img1.unsqueeze_(dim=0)  # expand for batch dim
        pred = self.model(img1)[0]

        # postprocess
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        classes = None  # not filter class_idx in results
        agnostic_nms = False
        max_det = 1000

        pred = non_max_suppression(pred,
                                   conf_thres,
                                   iou_thres,
                                   classes,
                                   agnostic_nms,
                                   max_det=max_det)

        result = []
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4],
                                          img.shape).round()
                result.append(det)

        # xyxy, conf, cls
        if len(result) > 0:
            tensor_result = torch.cat(result, dim=0)
            numpy_result = tensor_result.data.cpu().numpy()
        else:
            numpy_result = np.zeros(shape=(0, 6), dtype=np.float32)

        return numpy_result

    def infer(self, img: CV_IMAGE) -> List[rw.Annotation]:
        anns = []
        result = self.predict(img)

        for i in range(result.shape[0]):
            xmin, ymin, xmax, ymax, conf, cls = result[i, :6].tolist()
            ann = rw.Annotation(class_name=self.class_names[int(cls)],
                                score=conf,
                                box=rw.Box(x=int(xmin),
                                           y=int(ymin),
                                           w=int(xmax - xmin),
                                           h=int(ymax - ymin)))

            anns.append(ann)

        return anns


def convert_ymir_to_yolov5(cfg: edict) -> None:
    """
    convert ymir format dataset to yolov5 format
    generate data.yaml for training/mining/infer
    """

    data = dict(path=cfg.ymir.output.root_dir,
                nc=len(cfg.param.class_names),
                names=cfg.param.class_names)
    for split, prefix in zip(['train', 'val'],
                             ['training', 'val']):
        src_file = getattr(cfg.ymir.input, f'{prefix}_index_file')
        if osp.exists(src_file):
            shutil.copy(src_file, f'{cfg.ymir.output.root_dir}/{split}.tsv')

        data[split] = osp.join(cfg.ymir.output.root_dir, f'{split}.tsv')

    with open(osp.join(cfg.ymir.output.root_dir, 'data.yaml'), 'w') as fw:
        fw.write(yaml.safe_dump(data))
