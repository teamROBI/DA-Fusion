from detectron2.utils.visualizer import visualize_ours
import cv2
import numpy as np
import imageio
import math
import os
import torch

from legacy_paths import OCID_PATH

ocid = True
name = 'result_2018-08-21-14-50-11.png'
# Example: an OCID sequence. Edit these to point at any OCID/OSD/BOSD sample.
_seq = os.path.join(OCID_PATH, 'ARID20', 'table', 'top', 'seq08')
anno_path = os.path.join(_seq, 'label', name)
rgb_path = os.path.join(_seq, 'rgb', name)

def find_bounding_box(annotation):
    rows = len(annotation)
    cols = len(annotation[0]) if rows > 0 else 0
    
    min_x = cols
    max_x = -1
    min_y = rows
    max_y = -1
    
    for y in range(rows):
        for x in range(cols):
            if annotation[y][x] == 1:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    
    if min_x == cols or max_x == -1 or min_y == rows or max_y == -1:
        return None  
    
    x = min_x
    y = min_y
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    
    return (x, y, w, h)

W, H = 640, 480
BACKGROUND_LABEL = 0
BG_LABELS = {}
BG_LABELS["floor"] = [0, 1]
BG_LABELS["table"] = [0, 1, 2]

rgb_img = cv2.imread(rgb_path)
rgb_img = cv2.resize(rgb_img, (W, H))   
# laod GT (annotation) anno: [H, W]
gt = imageio.imread(anno_path)
gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

if ocid:
    floor_table = "floor" if f"{os.sep}floor{os.sep}" in rgb_path else "table"
    print(floor_table)
    for label in BG_LABELS[floor_table]:
        gt[gt == label] = 0 

labels_gt = np.unique(gt)
labels_gt = labels_gt[~np.isin(labels_gt, [BACKGROUND_LABEL])]
num_labels_gt = labels_gt.shape[0]

masks = []
bboxes = []
for i, gt_i in enumerate(labels_gt):
    gt_i_mask = (gt == gt_i)
    masks.append(gt_i_mask)
    bounding_box = find_bounding_box(gt_i_mask)
    bboxes.append(bounding_box)
    
masks = np.array(masks)
bboxes = np.array(bboxes)

output = visualize_ours(rgb_img, masks, bboxes)
image = output[:, :, ::-1]
# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
cv2.imwrite(f'{name}_GT.png', output)