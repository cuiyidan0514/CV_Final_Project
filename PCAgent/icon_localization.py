from PCAgent.crop import calculate_size, calculate_iou
from modelscope.pipelines import pipeline
from PIL import Image
import torch

def remove_boxes(boxes_filt, confidences_filt, labels_filt, size, iou_threshold=0.5):
    boxes_to_remove = set()

    for i in range(len(boxes_filt)):
        if calculate_size(boxes_filt[i]) > 0.05*size[0]*size[1]:
            boxes_to_remove.add(i)
        for j in range(len(boxes_filt)):
            if calculate_size(boxes_filt[j]) > 0.05*size[0]*size[1]:
                boxes_to_remove.add(j)
            if i == j:
                continue
            if i in boxes_to_remove or j in boxes_to_remove:
                continue
            iou = calculate_iou(boxes_filt[i], boxes_filt[j])
            if iou >= iou_threshold:
                boxes_to_remove.add(j)

    boxes_filt = [box for idx, box in enumerate(boxes_filt) if idx not in boxes_to_remove]
    confidences_filt = [confidence for idx, confidence in enumerate(confidences_filt) if idx not in boxes_to_remove]
    labels_filt = [label for idx, label in enumerate(labels_filt) if idx not in boxes_to_remove]
    
    return boxes_filt, confidences_filt, labels_filt


def det(input_image_path, caption, groundingdino_model, box_threshold=0.05, text_threshold=0.5, scale=1):
    # 在给定的输入图像中，检测指定caption相关的区域
    image = Image.open(input_image_path)
    size = image.size
    # print(size)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    
    # 格式化输入
    inputs = {
        'IMAGE_PATH': input_image_path,
        'TEXT_PROMPT': caption,
        'BOX_TRESHOLD': box_threshold,
        'TEXT_TRESHOLD': text_threshold
    }

    # 调用groundingdino模型，输入字典，获得检测框
    result = groundingdino_model(inputs) 
    bboxes = result['boxes']
    confidences = result['logits']
    labels = result['phrases']

    boxes_filt = []
    confidences_filt = []
    labels_filt = []
    for bbox, confidence, label in zip(bboxes, confidences, labels):
        if label:
            boxes_filt.append(bbox)
            confidences_filt.append(confidence)
            labels_filt.append(label)

    # print("Detected BBoxes:", boxes_filt)  
    # print("Detected Confidences:", confidences_filt)  
    # print("Detected Labels:", labels_filt)

    H, W = size[1], size[0]
    # 遍历每个预测框，将框的坐标按照原图的尺寸进行调整，转换为左上角和右下角坐标
    for i in range(len(boxes_filt)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = [box.cpu().int().tolist() for box in boxes_filt]
    # 过滤掉不合适的框，只保留前9个框
    filtered_boxes, filtered_confidences, filtered_labels = remove_boxes(boxes_filt, confidences_filt, labels_filt, size)  # [:9]
    print("Filtered BBoxes:", filtered_boxes)  
    print("Filtered Confidences:", filtered_confidences)  
    print("Filtered Labels:", filtered_labels)

    coordinates = []
    for box in filtered_boxes:
        b0 = int(box[0]/scale)
        b1 = int(box[1]/scale)
        b2 = int(box[2]/scale)
        b3 = int(box[3]/scale)
        coordinates.append([b0,b1,b2,b3])

    # 返回最终若干预测框的坐标
    return coordinates, filtered_confidences, filtered_labels