import os
import numpy as np
import cv2


def _binarize(preds):
    preds = preds.copy()
    preds[np.where(preds != 0)] = 1
    return preds


def post_image_process(preds, bboxes, w, h, rgb_img, visualize=True, area_cm2_map=None):
    """Post-process predicted masks. Ablatable via DAFUSION_POSTPROC:
        full (default) = merge_overlaps + refined_mask (original behavior)
        no_merge       = refined_mask only
        no_refine      = merge_overlaps only
        none           = neither (just prune <=10px + binarize) -- closest to the baselines
    merge_overlaps can under-segment clutter (unions overlapping masks); refined_mask can
    drop large/close objects (>40000px) causing misses. This toggle isolates their effect.

    `area_cm2_map` is an optional (H,W) map of each pixel's physical area in cm^2 (from depth);
    supplying it lets refined_mask gate on real object size instead of pixel count. See
    DAFUSION_METRIC_SIZE_GATE in `refined_mask`."""
    mode = os.environ.get("DAFUSION_POSTPROC", "full")
    # always prune specks (<=10 px)
    preds_len = [len(np.where(pred != 0)[0]) for pred in preds]
    prune_idx = np.where(np.array(preds_len) > 10)
    preds = preds[prune_idx]
    bboxes = bboxes[prune_idx]
    if len(preds) == 0 or mode == "none":
        return _binarize(preds), bboxes

    if mode != "no_merge":
        preds_len = [len(np.where(pred != 0)[0]) for pred in preds]
        preds, bboxes = merge_overlaps(preds, preds_len, bboxes, w, h, visualize=False)
    else:
        preds = _binarize(preds)

    if mode != "no_refine" and len(preds) > 0:
        preds, bboxes, _ = refined_mask(preds, bboxes, w, h, rgb_img, visualize=False,
                                        area_cm2_map=area_cm2_map)

    return preds, bboxes

def prune_invalid(preds):
    prune_keys = []
    for idx, pred in enumerate(preds):
        mask = np.ascontiguousarray(preds[idx], dtype=np.uint8)

        contours = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        max_cnt = max(contours, key=cv2.contourArea, default=0)  # max contour point
        if cv2.contourArea(max_cnt) > 50000 or cv2.contourArea(max_cnt) < 2000:
            prune_keys.append(idx)
            # print("too large or small", cv2.contourArea(max_cnt))

    return np.delete(preds, prune_keys, axis=0)


def merge_overlaps(preds, preds_len, bboxes, w, h, visualize=False):
    merged = []
    splited_pred = []
    for i in range(len(preds)):
        if i not in merged:
            for j in range(i + 1, len(preds)):
                if j not in merged:
                    overlapped = len(np.where(np.logical_and(preds[i] != 0, preds[j] != 0))[0])
                    small_overlap = overlapped / np.min([preds_len[i], preds_len[j]])
                    large_overlap = overlapped / np.max([preds_len[i], preds_len[j]])
                    if 0.7 < small_overlap:
                        # print(small_overlap, large_overlap, overlapped, preds_len[i], preds_len[j])
                        # preds[i] = np.mean(np.concatenate(([preds[i]], [preds[j]]), axis=0), axis=0)
                        preds[i] += preds[j]
                        preds_len[i] = len(np.where(preds[i] != 0)[0])
                        merged.append(j)

            # max_merged = np.max(preds[i][np.where(preds[i] != 0)])
            # print("max merged:", max_merged)
            if visualize:
                cv2.imshow("overlapped check", normalize_depth(preds[i], min_val=0, max_val=max_merged))
                key = cv2.waitKey(0)
                if key == 27:
                    break

            # if max_merged > 10:
            #     split1, split2 = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
            #     split1[np.where(preds[i] >= 10)] = 1
            #     split2[np.where(preds[i] < 10)] = 1
            #     splited_pred.append(split1)
            #     splited_pred.append(split2)

    preds = np.delete(preds, merged, axis=0)
    bboxes = np.delete(bboxes, merged, axis=0)
    preds[np.where(preds != 0)] = 1
    # print(preds.shape, np.array([split1]).shape)
    # preds = np.concatenate((preds, [split1], [split2]), axis=0)

    return preds, bboxes


def refined_mask(preds, bboxes, w, h, rgb_img, visualize=False, area_cm2_map=None):
    """Drop masks outside a size band, then prune each survivor to one compact blob.

    SIZE GATE. By default the band is in PIXELS ([500, 40000] on the largest contour). Pixel area
    is not a property of the object -- it falls as 1/z^2 and scales with image resolution -- so the
    same band is a different physical criterion on each benchmark. Measured on GT
    (`scripts/measure_gt_metric_size.py`), the 40000 px upper bound corresponds to 1383 cm^2 on
    OCID but only 598 cm^2 on OCBD (2.3x tighter), and the band deletes 5.16% of OCBD GT objects
    versus 0.58% on OCID and 0.00% on OSD.

    Setting DAFUSION_METRIC_SIZE_GATE="lo,hi" (cm^2) switches the gate to physical area, which is
    the same criterion everywhere. It needs `area_cm2_map` (per-pixel cm^2 from depth); without it
    the pixel band is used unchanged. Pooled GT spans p1=16 / p50=88 / p99=753 cm^2, so "5,2000"
    retains 99.65% of GT objects on all three sets.

    FRAGMENT PRUNING (contour >150 px, centre within 100 px) is left in pixels deliberately: it
    removes speckle artifacts of the mask raster itself, not objects, and measures as harmless --
    mean IoU(refined(GT), GT) = 0.9985 with 0.53% of objects losing >10% of area.
    """
    add_keys = []
    max_center_list = []
    max_area = 40000
    min_area = 500
    # Band syntax: "lo,hi" in cm^2. Either side may be the literal "px" to keep that end in pixels,
    # e.g. "px,2000" = pixel lower bound + 2000 cm^2 upper bound. Needed because the two bounds fail
    # for opposite reasons and a single-unit band confounds them: pixel-weighted metric accounting
    # says OCBD's 14 too-LARGE real objects are 3.77% of GT pixels while its 63 too-small ones are
    # 0.14%, so the upper bound is where nearly all the recoverable recall lives -- but loosening it
    # also lets large spurious predictions survive, and precision is pixel-weighted too.
    metric_band = os.environ.get("DAFUSION_METRIC_SIZE_GATE")
    min_cm2 = max_cm2 = None
    if metric_band and area_cm2_map is not None:
        lo_s, hi_s = metric_band.split(",")
        min_cm2 = None if lo_s.strip() == "px" else float(lo_s)
        max_cm2 = None if hi_s.strip() == "px" else float(hi_s)
        if min_cm2 is None and max_cm2 is None:
            metric_band = None
    else:
        metric_band = None
    cnt_prune_list = []
    cnt_remain_list = []

    for idx, pred in enumerate(preds):
        prune_mask = np.zeros(pred.shape[:2], dtype=np.uint8)
        orignal_mask = np.ascontiguousarray(preds[idx], dtype=np.uint8)

        contours = cv2.findContours(orignal_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        max_cnt = max(contours, key=cv2.contourArea, default=0) # max contour point
        x, y, w, h = cv2.boundingRect(max_cnt)
        max_center = np.array([x + w / 2, y + h / 2])
        if metric_band is not None:
            # physical area of the largest component: sum per-pixel cm^2 inside that contour.
            # Pixels with no depth are excluded and compensated by the valid fraction, since
            # OCID/OSD carry 12-24% holes.
            comp = np.zeros(orignal_mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(comp, [max_cnt], 0, 1, -1)
            comp = (comp > 0) & (orignal_mask > 0)
            npix = int(comp.sum())
            good = comp & (area_cm2_map > 0)
            nv = int(good.sum())
            a_cm2 = (float(area_cm2_map[good].sum()) * npix / nv) if nv else 0.0
            # no depth anywhere in the mask -> physical size unknown, fall back to pixels
            if nv == 0:
                out_of_band = (cv2.contourArea(max_cnt) > max_area or
                               cv2.contourArea(max_cnt) < min_area)
            else:
                out_of_band = ((max_cm2 is not None and a_cm2 > max_cm2) or
                               (min_cm2 is not None and a_cm2 < min_cm2) or
                               # a side left as "px" keeps the original pixel bound
                               (max_cm2 is None and cv2.contourArea(max_cnt) > max_area) or
                               (min_cm2 is None and cv2.contourArea(max_cnt) < min_area))
        else:
            out_of_band = (cv2.contourArea(max_cnt) > max_area or
                           cv2.contourArea(max_cnt) < min_area)
        if out_of_band:
            # print("too large or small", cv2.contourArea(max_cnt))
            cnt_prune_list.append(cv2.contourArea(max_cnt))
            continue
        else:
            cnt_dists = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 150:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cnt_center = np.array([x + w / 2, y + h / 2])
                    cnt_dists.append(np.linalg.norm(max_center - cnt_center))
                    if cnt_dists[-1] < 100:
                        cv2.drawContours(prune_mask, [cnt], 0, (1), -1)
                        # cv2.fillPoly(back, [cnt], (255, 255, 255))
            prune_mask = cv2.bitwise_and(orignal_mask, orignal_mask, mask=prune_mask) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            morphclose = cv2.morphologyEx(prune_mask, cv2.MORPH_CLOSE, kernel)

            preds[idx] = prune_mask.astype(bool)
            add_keys.append(idx)
            max_center_list.append(max_center)
            cnt_remain_list.append(cv2.contourArea(max_cnt))

            if visualize:
                print("contour size:", [cv2.contourArea(c) for c in contours], "contour dist from max:", cnt_dists)
                vis_img = np.hstack([orignal_mask*255, prune_mask, morphclose])
                cv2.imshow('Mask post processed', vis_img)

                back_rgb = cv2.cvtColor(prune_mask, cv2.COLOR_GRAY2RGB)
                back_rgb[np.where(prune_mask == 255)] = [0, 0, 255]
                rgb_img = cv2.addWeighted(rgb_img, 1, back_rgb, 0.8, 0.5)
                cv2.imshow('RGB Image', rgb_img)
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()
                    break
    
    if visualize:
        print(f"Contour threshold: {min_area}, {max_area}")
        print(f"Pruned list: {cnt_prune_list}")
        print(f"Remained list: {cnt_remain_list}")

    return preds[add_keys], bboxes[add_keys], max_center_list


def group_attention(preds,bbox, max_center_list):
    prune_keys = []
    for i in range(len(max_center_list)):
        dist = []
        for j in range(len(max_center_list)):
            if i != j:
                dist.append(np.linalg.norm(max_center_list[i] - max_center_list[j]))
        if np.min(dist) > 150:
            prune_keys.append(i)
    #     print(np.min(dist))
    # print(prune_keys)
    return np.delete(preds, prune_keys, axis=0), np.delete(bbox, prune_keys, axis=0)


def average_filter(rgb_img, depth_img, kernel_size):
    # kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    # Pad the image to handle border pixels
    padding = kernel_size // 2
    padded_image = np.pad(rgb_img, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    padded_kernel = np.pad(depth_img, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    filtered_image = np.zeros_like(rgb_img, dtype=np.uint8)

    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            for c in range(rgb_img.shape[2]):
                # Apply the kernel to each channel of the image
                neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                kernel = padded_kernel[i:i + kernel_size, j:j + kernel_size, c]
                # kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
                kernel = np.where((kernel[1, 1] - 5 < kernel) & (kernel < kernel[1, 1] + 5), kernel, 0)
                kernel = kernel / np.sum(kernel)
                # print(kernel)
                filtered_image[i, j, c] = np.sum(neighborhood * kernel)

    return filtered_image
