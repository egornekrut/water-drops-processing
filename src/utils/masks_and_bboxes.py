from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import masks_to_boxes


def get_bbox_from_mask(full_mask: np.ndarray, divider: int = 32) -> Tuple[int, int, int, int]:
    bbox = masks_to_boxes(torch.from_numpy(full_mask).unsqueeze(0))[0].tolist()
    return make_bbox_divided(bbox, divider)


def make_bbox_divided(bbox: Tuple[int, int, int, int], divider: int = 32) -> Tuple[int, int, int, int]:
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]

    new_h = height + divider - height % divider
    new_w = width + divider - width % divider
    offset_h = new_h - height
    offset_w = new_w - width

    return bbox[0] - offset_h // 2, bbox[1] - offset_w // 2, bbox[2] + offset_h // 2 + offset_h % 2,  bbox[3] + offset_w // 2 + offset_w % 2


def xywh_xyxy(box, img_size):
    return ((box[0] - box[2] / 2) * img_size[0], (box[1] - box[3] / 2) * img_size[1], (box[0] + box[2] / 2) * img_size[0], (box[1] + box[3] / 2) * img_size[1])


def determine_bboxes(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_wh = mask.shape

    all_bboxes = []

    for i in range(len(contours)):
        mined = contours[i][:, 0, :].min(0)
        maxed = contours[i][:, 0, :].max(0)
        bbox_coords = (mined[0], mined[1], maxed[0], maxed[1])

        if bbox_coords[2] - bbox_coords[0] < 2 or bbox_coords[3] - bbox_coords[1] < 2:
            continue
        shape_norm = (
            (bbox_coords[2] + bbox_coords[0]) / (2 * img_wh[0]),
            (bbox_coords[3] + bbox_coords[1]) / (2 * img_wh[1]),
            (bbox_coords[2] - bbox_coords[0]) / img_wh[0],
            (bbox_coords[3] - bbox_coords[1]) / img_wh[1],
        )
        all_bboxes.append(shape_norm)

    return all_bboxes
