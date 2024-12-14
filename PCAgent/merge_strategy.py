import numpy as np


def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1: list or array [x1, y1, x2, y2]
    - box2: list or array [x1, y1, x2, y2]

    Returns:
    - iou: float, IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # print(x2_inter, x1_inter, y2_inter, y1_inter)

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou


def merge_boxes(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    merged_box = [min(x1_min, x2_min), min(y1_min, y2_min), max(x1_max, x2_max), max(y1_max, y2_max)]
    return merged_box


def merge_boxes_and_texts(texts, boxes, iou_threshold=0):
    """
    Merge bounding boxes and their corresponding texts based on IoU threshold.

    Parameters:
    - boxes: List of bounding boxes, with each box represented as [x1, y1, x2, y2].
    - texts: List of texts corresponding to each bounding box.
    - iou_threshold: Intersection-over-Union threshold for merging boxes.

    Returns:
    - merged_boxes: List of merged bounding boxes.
    - merged_texts: List of merged texts corresponding to the bounding boxes.
    """
    if len(boxes) == 0:
        return [], []

    # boxes = np.array(boxes)
    merged_boxes = []
    merged_texts = []

    while len(boxes) > 0:
        box = boxes[0]
        text = texts[0]
        boxes = boxes[1:]
        texts = texts[1:]
        to_merge_boxes = [box]
        to_merge_texts = [text]
        keep_boxes = []
        keep_texts = []

        for i, other_box in enumerate(boxes):
            if compute_iou(box, other_box) > iou_threshold:
                to_merge_boxes.append(other_box)
                to_merge_texts.append(texts[i])
            else:
                keep_boxes.append(other_box)
                keep_texts.append(texts[i])

        # Merge the to_merge boxes into a single box
        if len(to_merge_boxes) > 1:
            x1 = min(b[0] for b in to_merge_boxes)
            y1 = min(b[1] for b in to_merge_boxes)
            x2 = max(b[2] for b in to_merge_boxes)
            y2 = max(b[3] for b in to_merge_boxes)
            merged_box = [x1, y1, x2, y2]
            merged_text = " ".join(to_merge_texts)  # You can change the merging strategy here
            merged_boxes.append(merged_box)
            merged_texts.append(merged_text)
        else:
            merged_boxes.extend(to_merge_boxes)
            merged_texts.extend(to_merge_texts)

        # boxes = np.array(keep_boxes)
        boxes = keep_boxes
        texts = keep_texts

    return merged_texts, merged_boxes

def calculate_distance(box, other_box):
    box_xmin, box_ymin, box_xmax, box_ymax = box  
    other_xmin, other_ymin, other_xmax, other_ymax = other_box

    if box_xmax < other_xmin:  # box 在 other_box 的左方  
        horizontal_distance = other_xmin - box_xmax  
    elif other_xmax < box_xmin:  # box 在 other_box 的右方  
        horizontal_distance = box_xmin - other_xmax  
    else:  
        # 他们在横向上重叠  
        horizontal_distance = 0  
    
    if box_ymax < other_ymin:  # box 在 other_box 的上方  
        vertical_distance = other_ymin - box_ymax  
    elif other_ymax < box_ymin:  # box 在 other_box 的下方  
        vertical_distance = box_ymin - other_ymax  
    else:  
        # 他们在纵向上重叠  
        vertical_distance = 0 
    
    return max(horizontal_distance, vertical_distance) 

def merge_buttons_and_texts(text_boxes, icon_boxes, icon_logits, icon_labels, icon_only=False, text_only=False):
    if len(icon_boxes) == 0:
        return []

    merged_boxes = []
    merged_logits = []
    merged_labels = []
    merged_text = set()

    for i in range(len(icon_boxes)):
        box = icon_boxes[i]
        to_merge_boxes = [box]
        min_distance = 5
        min_index = -1
        # 检测是否有相近的文本框
        for j, text_box in enumerate(text_boxes):
            distance = calculate_distance(text_box, box)
            if distance < min_distance:
                min_distance = distance
                min_index = j
        merged_text.add(j)
        # 如果有，则将icon和text的bbox合并
        if min_index != -1 and min_distance < 5:
            to_merge_boxes.append(text_boxes[min_index])
            x1 = min(b[0] for b in to_merge_boxes)
            y1 = min(b[1] for b in to_merge_boxes)
            x2 = max(b[2] for b in to_merge_boxes)
            y2 = max(b[3] for b in to_merge_boxes)
            merged_box = [x1, y1, x2, y2]
            merged_boxes.append(merged_box)
            merged_logits.append(icon_logits[i])
            merged_labels.append(icon_labels[i])
        # 如果没有，就只添加icon bbox
        elif icon_only:
            merged_boxes.append(icon_boxes[i])
            merged_logits.append(icon_logits[i])
            merged_labels.append(icon_labels[i])
        
    if text_only:
        for j, text_box in enumerate(text_boxes):
            if j in merged_text:
                continue
            merged_boxes.append(text_boxes[j])
            merged_logits.append(1)
            merged_labels.append("text")
            
    return merged_boxes, merged_logits, merged_labels


def merge_texts_and_icons(text_boxes, icon_boxes, icon_logits, icon_labels, icon_only=False, text_only=False):
    if len(icon_boxes) == 0:
        return []

    merged_boxes = []
    merged_logits = []
    merged_labels = []
    merged_icons = set()

    for i in range(len(text_boxes)):
        box = text_boxes[i]
        to_merge_boxes = [box]
        min_distance = 5
        min_index = -1
        # 检测是否有相近的文本框
        for j, icon_box in enumerate(icon_boxes):
            distance = calculate_distance(icon_box, box)
            if distance < min_distance:
                min_distance = distance
                min_index = j
        merged_icons.add(j)
        # 如果有，则将icon和text的bbox合并
        if min_index != -1 and min_distance < 5:
            to_merge_boxes.append(icon_boxes[min_index])
            x1 = min(b[0] for b in to_merge_boxes)
            y1 = min(b[1] for b in to_merge_boxes)
            x2 = max(b[2] for b in to_merge_boxes)
            y2 = max(b[3] for b in to_merge_boxes)
            merged_box = [x1, y1, x2, y2]
            merged_boxes.append(merged_box)
            merged_logits.append(icon_logits[min_index])
            merged_labels.append(icon_labels[min_index])
        # 如果没有，就只添加icon bbox
        elif text_only:
            merged_boxes.append(text_boxes[i])
            merged_logits.append(1)
            merged_labels.append("text")
        
    if icon_only:
        for j, icon_box in enumerate(icon_boxes):
            if j in merged_icons:
                continue
            merged_boxes.append(icon_boxes[j])
            merged_logits.append(icon_logits[j])
            merged_labels.append(icon_labels[j])
            
    return merged_boxes, merged_logits, merged_labels



def is_contained(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    if (x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max):
        return True
    elif (x2_min >= x1_min and y2_min >= y1_min and x2_max <= x1_max and y2_max <= y1_max):
        return True
    return False


def is_overlapping(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
        return True
    return False


def get_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min)

def merge_all_boxes_on_logits(cur_coords, cur_logits, cur_labels, pre_coords, pre_logits, pre_labels):
    result_bboxes, result_cons, result_labels = [],[],[]
    merged_cur_indices = set()
    
    for i,(pre_coord,pre_logit,pre_label) in enumerate(zip(pre_coords, pre_logits, pre_labels)):
        #print("ori_bbox:",pre_coord,pre_logit,pre_label)
        for j,(coord,logit,label) in enumerate(zip(cur_coords, cur_logits, cur_labels)):
            if j in merged_cur_indices:
                continue

            iou = calculate_iou(pre_coord,coord)
            if iou > 0.8:
                #print("bbox to merge:",coord,logit,label)
                merged_cur_indices.add(j)

                if logit > pre_logit:
                    max_logit = logit
                    max_label = label
                else:
                    max_logit = pre_logit
                    max_label = pre_label                
                
                x1_min, y1_min, x1_max, y1_max = coord
                x2_min, y2_min, x2_max, y2_max = pre_coord
                x1 = min(x1_min,x2_min)
                y1 = min(y1_min,y2_min)
                x2 = max(x1_max,x2_max)
                y2 = max(y1_max,y2_max)
                new_coord = [x1,y1,x2,y2]
                result_bboxes.append(new_coord)
                result_cons.append(max_logit)
                result_labels.append(max_label)
                #print("new bbox:",new_coord,max_logit,max_label)
                break
        
        # 如果遍历完pre_coords, pre_logits, pre_labels都没有重叠的，就加入result数组
        result_bboxes.append(pre_coord)
        result_cons.append(pre_logit)
        result_labels.append(pre_label)
    
    # 全部遍历完后，如果pre_coords, pre_logits, pre_labels还有剩下的，即没有和原来重叠的，则加入result
    for j,(coord,logit,label) in enumerate(zip(cur_coords, cur_logits, cur_labels)):
        if j not in merged_cur_indices:
            result_bboxes.append(coord)
            result_cons.append(logit)
            result_labels.append(label)       

    return result_bboxes, result_cons, result_labels

def merge_min_boxes(cur_coords, pre_coords):
    # 如果有重叠就取较小框，没有重叠就舍弃
    result_bboxes = []
    merged_cur_indices = set()
    
    for i,pre_coord in enumerate(pre_coords):
        for j,coord in enumerate(cur_coords):
            # if j in merged_cur_indices:
            #     continue
            iou = calculate_iou(pre_coord,coord)
            if iou > 0.3:
                # merged_cur_indices.add(j)

                x1_min, y1_min, x1_max, y1_max = coord
                x2_min, y2_min, x2_max, y2_max = pre_coord
                x1 = max(x1_min,x2_min)
                y1 = max(y1_min,y2_min)
                x2 = min(x1_max,x2_max)
                y2 = min(y1_max,y2_max)
                new_coord = [x1,y1,x2,y2]
                result_bboxes.append(new_coord)
                # break
    return result_bboxes


def merge_all_icon_boxes(bboxes,confidences,labels):
    result_bboxes = []
    result_cons = []
    result_labels = []
    while bboxes:
        bbox = bboxes.pop(0)
        confidence = confidences.pop(0)
        label = labels.pop(0)
        to_add = True

        for idx, existing_bbox in enumerate(result_bboxes):
            if is_contained(bbox, existing_bbox):
                if get_area(bbox) > get_area(existing_bbox):
                    result_bboxes[idx] = existing_bbox
                to_add = False
                break
            elif is_overlapping(bbox, existing_bbox) and calculate_iou(bbox, existing_bbox) > 1e-2:
                #print(calculate_iou(bbox, existing_bbox))
                if get_area(bbox) < get_area(existing_bbox):
                    result_bboxes[idx] = bbox
                to_add = False
                break

        if to_add:
            result_bboxes.append(bbox)
            result_cons.append(confidence)
            result_labels.append(label)

    return result_bboxes, result_cons, result_labels


def merge_bbox_groups(A, B, iou_threshold=0.8):
    i = 0
    while i < len(A):
        box_a = A[i]
        has_merged = False
        for j in range(len(B)):
            box_b = B[j]
            iou = calculate_iou(box_a, box_b)
            if iou > iou_threshold:
                merged_box = merge_boxes(box_a, box_b)
                A[i] = merged_box
                B.pop(j)
                has_merged = True
                break

        if has_merged:
            i -= 1
        i += 1

    return A, B


def bbox_iou(boxA, boxB):
    # Calculate Intersection over Union (IoU) between two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def merge_boxes_and_texts_new(texts, bounding_boxes, iou_threshold=0):
    if not bounding_boxes:
        return [], []

    bounding_boxes = np.array(bounding_boxes)
    merged_boxes = []
    merged_texts = []

    used = np.zeros(len(bounding_boxes), dtype=bool)

    for i, boxA in enumerate(bounding_boxes):
        if used[i]:
            continue
        x_min, y_min, x_max, y_max = boxA
        # text = texts[i]
        text = ''

        overlapping_indices = [i] # []
        for j, boxB in enumerate(bounding_boxes):
            # print(i,j, bbox_iou(boxA, boxB))
            if i != j and not used[j] and bbox_iou(boxA, boxB) > iou_threshold:
                overlapping_indices.append(j)

        # Sort overlapping boxes by vertical position (top to bottom)
        overlapping_indices.sort(key=lambda idx: (bounding_boxes[idx][1] + bounding_boxes[idx][3])/2) # TODO

        for idx in overlapping_indices:
            boxB = bounding_boxes[idx]
            x_min = min(x_min, boxB[0])
            y_min = min(y_min, boxB[1])
            x_max = max(x_max, boxB[2])
            y_max = max(y_max, boxB[3])
            # text += " " + texts[idx]
            text += texts[idx]
            used[idx] = True

        merged_boxes.append([x_min, y_min, x_max, y_max])
        merged_texts.append(text)
        used[i] = True

    return merged_texts, merged_boxes
