import numpy as np
import cv2 

def post_image_process(preds, bboxes, w, h, rgb_img, visualize=True):
    # print("Original preds len:", len(preds))
    preds_len = [len(np.where(pred != 0)[0]) for pred in preds]
    prune_idx = np.where(np.array(preds_len) > 10)
    preds = preds[prune_idx]
    bboxes = bboxes[prune_idx]
    # preds = prune_invalid(preds)
    # print("Delete nan preds len:", len(preds))

    preds_len = [len(np.where(pred != 0)[0]) for pred in preds]
    preds, bboxes = merge_overlaps(preds, preds_len, bboxes, w, h, visualize=False)
    # print("Overlap merged preds len:", len(preds))

    preds, bboxes, max_center_list = refined_mask(preds, bboxes, w, h, rgb_img, visualize=False)
    # print("Refined nan preds len:", len(preds))

    # preds, bboxes = group_attention(preds, bboxes, max_center_list)
    # print("Group attention preds len:", len(preds))

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


def refined_mask(preds, bboxes, w, h, rgb_img, visualize=False):
    add_keys = []
    max_center_list = []
    max_area = 40000
    min_area = 500
    cnt_prune_list = []
    cnt_remain_list = []
    
    for idx, pred in enumerate(preds):
        prune_mask = np.zeros(pred.shape[:2], dtype=np.uint8)
        orignal_mask = np.ascontiguousarray(preds[idx], dtype=np.uint8)

        contours = cv2.findContours(orignal_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        max_cnt = max(contours, key=cv2.contourArea, default=0) # max contour point
        x, y, w, h = cv2.boundingRect(max_cnt)
        max_center = np.array([x + w / 2, y + h / 2])
        if cv2.contourArea(max_cnt) > max_area or cv2.contourArea(max_cnt) < min_area:
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
