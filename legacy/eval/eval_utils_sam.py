from legacy_paths import SAM_WEIGHTS, CGNET_WEIGHTS, OSD_PATH, OCID_PATH, BOSD_PATH
import sys

import os
import cv2
import glob
import numpy as np
import imageio
import torch
from torchvision.ops import boxes as box_ops
import random

from tqdm import tqdm
from mask2former import add_maskformer2_config
import eval.post_process as pp
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from eval import compute_PRF 
from eval.model import Context_Guided_Network
from termcolor import colored

BACKGROUND_LABEL = 0
BG_LABELS = {}
BG_LABELS["floor"] = [0, 1]
BG_LABELS["table"] = [0, 1, 2]

def sam_output_to_instances(sam_outputs, image_size):
    # 마스크, 바운딩 박스, 신뢰도 리스트 초기화
    masks = []
    boxes = []
    
    for output in sam_outputs:
        masks.append(torch.tensor(output['segmentation'], dtype=torch.uint8))
        boxes.append(torch.tensor(output['bbox'], dtype=torch.float32))
    
    # Mask2Former나 Mask R-CNN의 Instances 형태에 맞게 변환
    boxes = box_ops.clip_boxes_to_image(torch.stack(boxes), image_size)
    masks = torch.stack(masks)
    
    # Instances 객체 구성 (사전에 정의된 객체 형태일 수 있음)
    instances = {
        'pred_boxes': boxes,
        'pred_masks': masks,
    }
    
    return instances

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



def eval_visible_on_OSD(write=True):
    sam_checkpoint = SAM_WEIGHTS
    model_type = "vit_h"
    device = "cuda"
    use_cgnet = True
    cgnet_weight_path = CGNET_WEIGHTS
    W, H = 640, 480
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    print(">>> Eval OSD")

    dataset_path = OSD_PATH
    # load dataset
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(dataset_path)))
    anno_paths = sorted(glob.glob("{}/annotation/*.png".format(dataset_path)))
    assert len(rgb_paths) == len(depth_paths)
    assert len(rgb_paths) == len(anno_paths)
    print(colored("Evaluation on OSD dataset: {} rgbs, {} depths, {} visible masks".format(
                len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))
    
    metrics_all = []
    iou_masks = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance
    
    
    # foreground segmentation
    if use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(rgb_paths), depth_paths, anno_paths)):
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))   
        depth_img = imageio.imread(depth_path).astype(np.float32)
        depth_img = normalize_depth(depth_img) # normalize_depth(depth, min_val=300.0, max_val=1800.0)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # laod GT (annotation) anno: [H, W]
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        num_inst_all += len(labels_anno)

        outputs = mask_generator_2.generate(rgb_img)
        image_size = (rgb_img.shape[0], rgb_img.shape[1])  # 이미지의 크기
        instances = sam_output_to_instances(outputs, image_size)
        pred_masks = instances['pred_masks'].detach().cpu().numpy()
        
        # CG-Net inference
        if use_cgnet:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)
            pred_all = np.zeros_like(anno)
            pred = np.zeros_like(anno)
            
            for i, mask in enumerate(pred_masks):
                # print(mask, fg_output.shape, type(mask), type(fg_output))
                iou = np.sum(np.bitwise_and(mask.astype(int), fg_output.astype(int))) / np.sum(mask)
                if iou >= 0.5:
                    pred[mask > False] = i+1
                pred_all[mask > False] = i+1
        else: 
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                pred[mask > False] = i+1
    
        # filter out the background instances
        if use_cgnet:
            remove_idxs = []
            for i, pred_mask in enumerate(pred_masks):
                pred_mask = pred_mask > 0
                fg_output = fg_output > 0
                iou = np.sum(np.bitwise_and(pred_mask, fg_output)) / np.sum(pred_mask)
                if iou < 0.5: 
                    remove_idxs.append(i)
            pred_masks = np.delete(pred_masks, remove_idxs, 0)

            
        # evaluate
        metrics, assignments = compute_PRF.multilabel_metrics(pred, anno, rgb_img, rgb_img, return_assign=True, save_img=False)
        
        metrics_all.append(metrics)

        # compute IoU for all instances
        # print(assignments)
        num_inst_mat += len(assignments)
        assign_visible_pred, assign_visible_gt = 0, 0
        assign_visible_overlap = 0
        for gt_id, pred_id in assignments:
            # count area of visible mask (pred & gt)
            mask_pr = pred == pred_id
            mask_gt = anno == gt_id           
            assign_visible_pred += np.count_nonzero(mask_pr)
            assign_visible_gt += np.count_nonzero(mask_gt)
            # count area of overlap btw. pred & gt
            mask_overlap = np.logical_and(mask_pr, mask_gt)
            assign_visible_overlap += np.count_nonzero(mask_overlap)
        if assign_visible_pred+assign_visible_gt-assign_visible_overlap > 0:
            iou = assign_visible_overlap / (assign_visible_pred+assign_visible_gt-assign_visible_overlap)
        else: iou = 0
        iou_masks += iou
    # compute mIoU for all instances
    miou = iou_masks / len(metrics_all)
    
    # sum the values with same keys
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num
    
    total_sum = (float(result['Objects Precision']) + float(result['Objects Recall']) + float(result['Objects F-measure']) +
                float(result['Boundary Precision']) + float(result['Boundary Recall']) + float(result['Boundary F-measure'])
                + float(result['obj_detected_075_percentage'])) * 100

    print('\n')
    print(colored("Visible Metrics for OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))

    # Write Result txt
    f = open('SAM_result.txt', 'a')
    f.write('\n===============================================\n')
    f.write('\n')
    f.write("Visible Metrics for OSD\n")
    f.write("---------------------------------------------\n")
    f.write("    Overlap    |    Boundary\n")
    f.write("  P    R    F  |   P    R    F  |  %75 | mIoU\n")
    f.write("---------------------------------------------\n")
    f.write("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}\n".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    f.write('\n')
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
        f.write('%s: %f\n' % (k, result[k]))
    print('\n')
    f.write('\n')
    f.close()



    return total_sum

def eval_visible_on_OCID():
    print(">>> Eval OCID")
    
    sam_checkpoint = SAM_WEIGHTS
    model_type = "vit_h"
    device = "cuda"
    W, H = 640, 480
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    
    mask_generator = SamAutomaticMaskGenerator(sam)

    dataset_path = OCID_PATH
    # load dataset
    image_paths = []
    depth_paths = []
    anno_paths = []
    # load ARID20
    print("... load dataset [ ARID20 ]")
    data_root = dataset_path + "/ARID20"
    f_or_t = ["floor", "table"]
    b_or_t = ["bottom", "top"]
    for dir_1 in f_or_t:
        for dir_2 in b_or_t:
            seq_list = sorted(os.listdir(os.path.join(data_root, dir_1, dir_2)))
            for seq in seq_list:
                data_dir = os.path.join(data_root, dir_1, dir_2, seq)
                if not os.path.isdir(data_dir): continue
                data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                for data_name in data_list:
                    image_path = os.path.join(data_root, dir_1, dir_2, seq, "rgb", data_name)
                    image_paths.append(image_path)
                    depth_path = os.path.join(data_root, dir_1, dir_2, seq, "depth", data_name)
                    depth_paths.append(depth_path)
                    anno_path = os.path.join(data_root, dir_1, dir_2, seq, "label", data_name)
                    anno_paths.append(anno_path)
    # load YCB10
    print("... load dataset [ YCB10 ]")
    data_root = dataset_path +  "/YCB10"
    f_or_t = ["floor", "table"]
    b_or_t = ["bottom", "top"]
    c_c_m = ["cuboid", "curved", "mixed"]
    for dir_1 in f_or_t:
        for dir_2 in b_or_t:
            for dir_3 in c_c_m:
                seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                for seq in seq_list:
                    data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                    if not os.path.isdir(data_dir): continue
                    data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                    for data_name in data_list:
                        image_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                        image_paths.append(image_path)
                        depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                        depth_paths.append(depth_path)
                        anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                        anno_paths.append(anno_path)
    # load ARID10
    print("... load dataset [ ARID10 ]")
    data_root =  dataset_path + "/ARID10"
    f_or_t = ["floor", "table"]
    b_or_t = ["bottom", "top"]
    c_c_m = ["box", "curved", "fruits", "mixed", "non-fruits"]
    for dir_1 in f_or_t:
        for dir_2 in b_or_t:
            for dir_3 in c_c_m:
                seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                for seq in seq_list:
                    data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                    if not os.path.isdir(data_dir): continue
                    data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                    for data_name in data_list:
                        image_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                        image_paths.append(image_path)
                        depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                        depth_paths.append(depth_path)
                        anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                        anno_paths.append(anno_path)
    assert len(image_paths) == len(depth_paths)
    assert len(image_paths) == len(anno_paths)
    print(colored("Evaluation on OCID dataset: {} rgbs, {} depths, {} visible_masks".format(
                    len(image_paths), len(depth_paths), len(anno_paths)), "green"))
    

    metrics_all = []
    ious_mask = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance

    
    for i, (rgb_path, anno_path) in enumerate(zip(tqdm(image_paths), anno_paths)):   
        rgb_img = cv2.imread(rgb_path)
        

        # load GT (annotation) anno: [H, W]
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)        
        # remove background, table
        floor_table = rgb_path.split("/")[9]
        for label in BG_LABELS[floor_table]:
            anno[anno == label] = 0         
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        num_inst_all += len(labels_anno)

        outputs = mask_generator_2.generate(rgb_img)
        image_size = (rgb_img.shape[0], rgb_img.shape[1])  # 이미지의 크기
        instances = sam_output_to_instances(outputs, image_size)
        pred_masks = instances['pred_masks'].detach().cpu().numpy()

        pred = np.zeros_like(anno)
        for i, mask in enumerate(pred_masks):
            pred[mask > False] = i+1
                

        # evaluate
        metrics, assignments = compute_PRF.multilabel_metrics(pred, anno, rgb_img, rgb_img, return_assign=True, save_img=False)

        metrics_all.append(metrics)

        # compute IoU for all instances
        # print(assignments)
        num_inst_mat += len(assignments)
        assign_visible_pred, assign_visible_gt = 0, 0
        assign_visible_overlap = 0
        for gt_id, pred_id in assignments:
            # count area of visible mask (pred & gt)
            mask_pr = pred == pred_id
            mask_gt = anno == gt_id           
            assign_visible_pred += np.count_nonzero(mask_pr)
            assign_visible_gt += np.count_nonzero(mask_gt)
            # count area of overlap btw. pred & gt
            mask_overlap = np.logical_and(mask_pr, mask_gt)
            assign_visible_overlap += np.count_nonzero(mask_overlap)
        if (assign_visible_pred+assign_visible_gt-assign_visible_overlap) > 0:
            iou = assign_visible_overlap / (assign_visible_pred+assign_visible_gt-assign_visible_overlap)
        else:
            iou = 0
        ious_mask += iou
    # compute mIoU for all instances
    miou = ious_mask / len(metrics_all)
    
    # sum the values with same keys
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Visible Metrics for OCID", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))

    # Write Result txt
    f = open('SAM_result.txt', 'a')
    f.write('\n===============================================\n')
    f.write('\n')
    f.write("Visible Metrics for OCID\n")
    f.write("---------------------------------------------\n")
    f.write("    Overlap    |    Boundary\n")
    f.write("  P    R    F  |   P    R    F  |  %75 | mIoU\n")
    f.write("---------------------------------------------\n")
    f.write("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}\n".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    f.write('\n')
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
        f.write('%s: %f\n' % (k, result[k]))
    print('\n')
    f.write('\n')
    f.close()

    return 


def eval_visible_on_BOSD():
    sam_checkpoint = SAM_WEIGHTS
    model_type = "vit_h"
    device = "cuda"
    W, H = 640, 480
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    
    mask_generator = SamAutomaticMaskGenerator(sam)

    dataset_path = BOSD_PATH
    # load dataset
    image_paths = []
    depth_paths = []
    anno_paths = []

    # load YCB
    print("... load dataset [ YCB ]")
    data_root = dataset_path + "/YCB"
    bin_list = ['scene_hole_gray_bin', 'scene_large_yellow_bin', 'scene_small_white_bin']
    for bin_name in bin_list:
        data_dir = os.path.join(data_root, bin_name)
        if not os.path.isdir(data_dir): continue
        data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
        for data_name in data_list:
            image_path = os.path.join(data_root, bin_name, "rgb", data_name)
            image_paths.append(image_path)
            depth_path = os.path.join(data_root, bin_name, "depth", data_name)
            depth_paths.append(depth_path)
            anno_path = os.path.join(data_root, bin_name, "label", data_name)
            anno_paths.append(anno_path)

    # load Non-YCB
    print("... load dataset [ Non-YCB ]")
    data_root = dataset_path + "/Non-YCB"
    bin_list = ['scene_hole_gray_bin', 'scene_large_yellow_bin', 'scene_small_white_bin']
    for bin_name in bin_list:
        data_dir = os.path.join(data_root, bin_name)
        if not os.path.isdir(data_dir): continue
        data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
        for data_name in data_list:
            image_path = os.path.join(data_root, bin_name, "rgb", data_name)
            image_paths.append(image_path)
            depth_path = os.path.join(data_root, bin_name, "depth", data_name)
            depth_paths.append(depth_path)
            anno_path = os.path.join(data_root, bin_name, "label", data_name)
            anno_paths.append(anno_path)
    
    metrics_all = []
    iou_masks = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance


    for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(image_paths), depth_paths, anno_paths)):
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))

        # laod GT (annotation) anno: [H, W]
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        num_inst_all += len(labels_anno)


        outputs = mask_generator_2.generate(rgb_img)
        image_size = (rgb_img.shape[0], rgb_img.shape[1])  # 이미지의 크기
        instances = sam_output_to_instances(outputs, image_size)
        pred_masks = instances['pred_masks'].detach().cpu().numpy()


        pred = np.zeros_like(anno)
        for i, mask in enumerate(pred_masks):
            pred[mask > False] = i+1
        
        # evaluate
        metrics, assignments = compute_PRF.multilabel_metrics(pred, anno, rgb_img, rgb_img, return_assign=True, save_img=False)

        metrics_all.append(metrics)

        # compute IoU for all instances
        # print(assignments)
        num_inst_mat += len(assignments)
        assign_visible_pred, assign_visible_gt = 0, 0
        assign_visible_overlap = 0
        for gt_id, pred_id in assignments:
            # count area of visible mask (pred & gt)
            mask_pr = pred == pred_id
            mask_gt = anno == gt_id           
            assign_visible_pred += np.count_nonzero(mask_pr)
            assign_visible_gt += np.count_nonzero(mask_gt)
            # count area of overlap btw. pred & gt
            mask_overlap = np.logical_and(mask_pr, mask_gt)
            assign_visible_overlap += np.count_nonzero(mask_overlap)
        if assign_visible_pred+assign_visible_gt-assign_visible_overlap > 0:
            iou = assign_visible_overlap / (assign_visible_pred+assign_visible_gt-assign_visible_overlap)
        else: iou = 0
        iou_masks += iou
    # compute mIoU for all instances
    miou = iou_masks / len(metrics_all)
    
    # sum the values with same keys
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num
        
    print('\n')
    print(colored("Visible Metrics for BOSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))

    # Write Result txt
    f = open('SAM_result.txt', 'a')
    f.write('\n===============================================\n')
    f.write('\n')
    f.write("Visible Metrics for BOSD\n")
    f.write("---------------------------------------------\n")
    f.write("    Overlap    |    Boundary\n")
    f.write("  P    R    F  |   P    R    F  |  %75 | mIoU\n")
    f.write("---------------------------------------------\n")
    f.write("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}\n".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    f.write('\n')
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
        f.write('%s: %f\n' % (k, result[k]))
    print('\n')
    f.write('\n')
    f.close()

    return