import argparse
import numpy as np
import os
import sys

import os
import cv2
import glob
import numpy as np
import imageio
import torch
import random

from tqdm import tqdm
from mask2former import add_maskformer2_config
from legacy_paths import BEST_CHECKPOINT, CGNET_WEIGHTS
from eval.model import Context_Guided_Network
import eval.post_process as pp
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer, visualize_ours
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.engine import (
    default_setup,
    launch,
)

def normalize_depth(depth, min_val=300.0, max_val=1800.0):
    min_val = max(np.percentile(depth, 5), min_val)
    max_val = min(np.percentile(depth, 95), max_val)
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))

    return depth

def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W // factor, H // factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)

    return depth


def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor

def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes
        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedul
    add_maskformer2_config(cfg)
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weight
    cfg.TEST.OVERLAP_THRESHOLD = 0.7
    cfg.TEST.OBJECT_MASK_THRESHOLD = 0.7
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    return cfg

def predict_img(args, rgb_path, depth_path=False):
    cfg = setup(args)
    input_type = args.input_type
    print(">>> Input type:", input_type)
    predictor = DefaultPredictor(cfg)
    W, H = 640, 480
    SCORE_THRESHOLD = 0.5
    
    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(CGNET_WEIGHTS)
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()
        
    if cfg.INPUT.DEPTH_INVERTED:
        print(">>> Depth Inverted")

        rgb_img = cv2.imread(rgb_path)
        # rgb_img = cv2.resize(rgb_img, (W, H))

        if depth_path:
            depth_img = imageio.imread(depth_path).astype(np.float32)
            depth_img = normalize_depth(depth_img) # normalize_depth(depth, min_val=300.0, max_val=1800.0)
            depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_img = inpaint_depth(depth_img)
            # depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)

        if depth_path and cfg.INPUT.DEPTH_INVERTED:
            depth_img = 255 - depth_img

        # load rgb and depth
        if input_type == 'rgb':
            our_input = rgb_img
        if input_type == 'depth':
            our_input = depth_img
        if input_type == 'rgbd':
            our_input = rgb_img
            our_depth_input = depth_img

        # forward (DualFormar)
        if input_type == "rgbd":
            outputs = predictor(our_input, our_depth_input)
        else:
            outputs = predictor(our_input)
        instances = outputs['instances'].to('cpu')
        filter_instances = instances[instances.scores > SCORE_THRESHOLD] # filter high score masks
        print(f"Masks above {SCORE_THRESHOLD}:", filter_instances.scores)

        pred_masks = filter_instances.pred_masks.detach().cpu().numpy()
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy()
        pred_masks, bboxes = pp.post_image_process(pred_masks, bboxes, W, H, rgb_img)

        # CG-Net inference
        if args.use_cgnet:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)

            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

    
        # filter out the background instances
        if args.use_cgnet:
            remove_idxs = []
            for i, pred_mask in enumerate(pred_masks):
                pred_mask = pred_mask > 0
                fg_output = fg_output > 0
                iou = np.sum(np.bitwise_and(pred_mask, fg_output)) / np.sum(pred_mask)
                if iou < 0.5: 
                    remove_idxs.append(i)
            pred_masks = np.delete(pred_masks, remove_idxs, 0)
            bboxes = np.delete(bboxes, remove_idxs, 0)
            
        vis_img = visualize_ours(rgb_img, pred_masks, bboxes)
        cv2.imwrite('test_result.png', vis_img)

if __name__ == "__main__":
    #python benchmark.py --input_type rgb --dataset ocid --gpu 2
    parser = argparse.ArgumentParser('Unseen Object Segmentation Benchmark Datasets', add_help=False)

    # model config
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--input_type", type=str, default="rgbd", help="rgb, depth, depth_colormap, rgbd")
    parser.add_argument("--use_cgnet", type=bool, default=False)
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="checkpoint folder (default: best checkpoint for --input_type)")
    parser.add_argument("--ckpt", type=str, default="model_final.pth", help="checkpoint filename inside exp_dir")
    parser.add_argument("--rgb_path", type=str, required=True, help="path to an RGB image")
    parser.add_argument("--depth_path", type=str, default=None, help="path to a depth image (for rgbd/depth)")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    exp_dir = args.exp_dir or BEST_CHECKPOINT.get(args.input_type)
    args.exp_name = os.path.basename(os.path.normpath(exp_dir))
    args.config_file = os.path.join(exp_dir, "config.yaml")
    args.weight = os.path.join(exp_dir, args.ckpt)
    predict_img(args, args.rgb_path, args.depth_path)