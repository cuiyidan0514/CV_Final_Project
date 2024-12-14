from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
# modelscope用于加载和使用深度学习模型
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download
import concurrent.futures
import os
# PCAgent包含自定义的图像处理和目标检测功能
from PCAgent.text_localization import ocr
from PCAgent.icon_localization import det
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes,merge_boxes_and_texts_new, merge_buttons_and_texts, merge_all_boxes_on_logits, merge_min_boxes, merge_texts_and_icons
from run_original import split_image_into_4,split_image_into_16,split_image_into_36,draw_coordinates_boxes_on_image,split_image_into_16_and_shift,split_image_into_25,text_draw_coordinates_boxes_on_image,split_image_into_9,draw_boxes_in_format
from detect_root_area import detect_root_area
import copy

import csv
import xml.etree.ElementTree as ET
import cv2
import pickle
from collections import Counter

class DetectionProcessor:
    def __init__(self):
        # 加载模型
        self.groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
        self.dino_model = pipeline('grounding-dino-task', model=self.groundingdino_dir)
        self.ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

    def split_image(self, image_path):
        """将图片分成4份以便并行处理"""
        img = Image.open(image_path)
        width, height = img.size
        sub_width = width // 2
        sub_height = height // 2
        parts = []
        coordinates = []
        for i in range(2):
            for j in range(2):
                box = (i*sub_width, j*sub_height, 
                      (i+1)*sub_width, (j+1)*sub_height)
                part = img.crop(box)
                part_path = f"temp_part_{i}_{j}.png"
                part.save(part_path)
                parts.append(part_path)
                coordinates.append((i*sub_width, j*sub_height))
        return parts, coordinates, (width, height)

    def process_part(self, part_path, offset_x, offset_y):
        """处理图片的一个部分"""
        # 目标检测
        boxes_dino = self.dino_model(part_path, "all objects")
        
        # OCR检测
        ocr_det = self.ocr_detection(part_path)
        ocr_boxes = []
        ocr_texts = []
        
        for box in ocr_det['polygons']:
            box = np.array(box).reshape(-1, 2)
            x1, y1 = box.min(axis=0)
            x2, y2 = box.max(axis=0)
            crop_img = Image.open(part_path).crop((x1, y1, x2, y2))
            text = self.ocr_recognition(crop_img)[0]['text']
            
            # 调整坐标到原图位置
            ocr_boxes.append([
                x1 + offset_x, y1 + offset_y,
                x2 + offset_x, y2 + offset_y
            ])
            ocr_texts.append(text)
        
        # 调整目标检测框的坐标
        adjusted_boxes = []
        for box in boxes_dino:
            adjusted_box = [
                box[0] + offset_x, box[1] + offset_y,
                box[2] + offset_x, box[3] + offset_y
            ]
            adjusted_boxes.append(adjusted_box)
            
        return {
            'dino_boxes': adjusted_boxes,
            'ocr_boxes': ocr_boxes,
            'ocr_texts': ocr_texts
        }

    def merge_results(self, results):
        """
        合并所有部分的结果：将所有部分的检测结果合并为一个完整的结果
        """
        all_dino_boxes = []
        all_ocr_boxes = []
        all_ocr_texts = []
        
        for result in results:
            all_dino_boxes.extend(result['dino_boxes'])
            all_ocr_boxes.extend(result['ocr_boxes'])
            all_ocr_texts.extend(result['ocr_texts'])
            
        return all_dino_boxes, all_ocr_boxes, all_ocr_texts

    def visualize_results(self, image_path, dino_boxes, ocr_boxes, ocr_texts, output_path):
        """可视化检测结果"""
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # 绘制目标检测框（红色）
        for i, box in enumerate(dino_boxes):
            draw.rectangle(box, outline=(255, 0, 0), width=2)
            draw.text((box[0], box[1]-15), f'obj_{i}', fill=(255, 0, 0))
            
        # 绘制OCR检测框（绿色）和文本
        for i, (box, text) in enumerate(zip(ocr_boxes, ocr_texts)):
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            draw.text((box[0], box[1]-15), text[:10], fill=(0, 255, 0))
            
        image.save(output_path)

    def process_image(self, image_path, output_path):
        """
        处理完整图片的主函数：
        包括分割、并行处理子图、合并子图结果、可视化最终检测
        """
        # 分割图片
        parts, coordinates, (width, height) = self.split_image(image_path)
        
        # 并行处理各个部分
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for part_path, (offset_x, offset_y) in zip(parts, coordinates):
                future = executor.submit(
                    self.process_part, 
                    part_path, 
                    offset_x, 
                    offset_y
                )
                futures.append(future)
            
            # 收集结果
            results = [future.result() for future in futures]
        
        # 合并结果
        dino_boxes, ocr_boxes, ocr_texts = self.merge_results(results)
        
        # 可视化结果
        self.visualize_results(image_path, dino_boxes, ocr_boxes, ocr_texts, output_path)
        
        # 清理临时文件
        for part_path in parts:
            os.remove(part_path)
            
        return {
            'dino_boxes': dino_boxes,
            'ocr_boxes': ocr_boxes,
            'ocr_texts': ocr_texts
        }


# 用于识别图像中的对象
groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
# 用于检测图像中的文字区域
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
# 用于识别文本内容
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

# 设置绘制参数
DRAW_TEXT_BOX = 0 # 先不考虑ocr
USE_SOM = 1
ICON_CAPTION = 1
LOCATION_INFO = 'center'
caption_call_method = "local"
# font_path = r'C:\Windows\Fonts\Arial.ttf' # windows
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" #! for linux
temp_file = "temp"


# 从原图中将root区域裁剪出来:基于gt
def extract_root_from_image(screenshot_file, xml_file, output_file, clear_ocr=False):
    if clear_ocr:
        output_file_name = f"{output_file}root_clear_ocr.png"
    else:
        output_file_name = f"{output_file}root.png"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == "root" or name == "Root":
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            with Image.open(screenshot_file) as img:
                cropped_img = img.crop((xmin,ymin,xmax,ymax))
                cropped_img.save(output_file_name)
                print(f'Saved cropped image to {output_file_name}')
            
            return xmin,ymin
        
def clear_det_from_img(merged_text_coordinates, screenshot_file, output_file, merged_logits=None, th=0):  
    image = cv2.imread(screenshot_file)  
    
    if image is None:  
        print("Error: Could not read the image.")  
        return  
    
    output_image = image.copy()  
    
    for (x1, y1, x2, y2),logit in zip(merged_text_coordinates, merged_logits):  
        if logit < th:
            continue
        # 计算 bbox 周围的区域  
        # 定义 ROI 的边界（包括扩展1个像素）  
        roi_x1 = max(x1 - 2, 0)  
        roi_y1 = max(y1 - 2, 0)  
        roi_x2 = min(x2 + 2, image.shape[1])  
        roi_y2 = min(y2 + 2, image.shape[0])  
        
        # 提取整个周围区域  
        complete_roi = output_image[roi_y1:roi_y2, roi_x1:roi_x2]  
        
        # 创建黑色掩码，仅保留 bbox 内的部分  
        mask = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1), dtype=np.uint8)  # Create a mask with zeros  
        cv2.rectangle(mask, (x1 - roi_x1, y1 - roi_y1), (x2 - roi_x1, y2 - roi_y1), 255, -1)  # Fill rectangle  

        # 从完整的 ROI 中排除 bbox 内的部分  
        masked_roi = cv2.bitwise_and(complete_roi, complete_roi, mask=cv2.bitwise_not(mask))  
        
        # 检查 masked_roi，不为空长度  
        if masked_roi.size > 0:  
            # 计算平均颜色，仅考虑周围颜色  
            avg_color = cv2.mean(masked_roi, mask=cv2.bitwise_not(mask))[0:3]  
        else:  
            avg_color = (0, 0, 0)  # 如果没有有效区域则设置为黑色  
        
        # 用平均颜色填充 bbox 区域  
        cv2.rectangle(output_image, (x1, y1), (x2, y2), tuple(map(int, avg_color)), -1)  

    # 保存结果图像  
    output_name = f"{output_file}.png"
    cv2.imwrite(output_name, output_image)  
    print(f"Output saved to: {output_name}")
    return output_name

        
def clear_ocr_from_img(merged_text_coordinates, screenshot_file, output_file):  
    image = cv2.imread(screenshot_file)  
    
    if image is None:  
        print("Error: Could not read the image.")  
        return  
    
    output_image = image.copy()  
    
    for (x1, y1, x2, y2) in merged_text_coordinates:  
        # 计算 bbox 周围的区域  
        # 定义 ROI 的边界（包括扩展1个像素）  
        roi_x1 = max(x1 - 1, 0)  
        roi_y1 = max(y1 - 1, 0)  
        roi_x2 = min(x2 + 1, image.shape[1])  
        roi_y2 = min(y2 + 1, image.shape[0])  
        
        # 提取整个周围区域  
        complete_roi = output_image[roi_y1:roi_y2, roi_x1:roi_x2]  
        
        # 创建黑色掩码，仅保留 bbox 内的部分  
        mask = np.zeros((roi_y2 - roi_y1, roi_x2 - roi_x1), dtype=np.uint8)  # Create a mask with zeros  
        cv2.rectangle(mask, (x1 - roi_x1, y1 - roi_y1), (x2 - roi_x1, y2 - roi_y1), 255, -1)  # Fill rectangle  

        # 从完整的 ROI 中排除 bbox 内的部分  
        masked_roi = cv2.bitwise_and(complete_roi, complete_roi, mask=cv2.bitwise_not(mask))  
        
        # 检查 masked_roi，不为空长度  
        if masked_roi.size > 0:  
            # 计算平均颜色，仅考虑周围颜色  
            avg_color = cv2.mean(masked_roi, mask=cv2.bitwise_not(mask))[0:3]  
        else:  
            avg_color = (0, 0, 0)  # 如果没有有效区域则设置为黑色  
        
        # 用平均颜色填充 bbox 区域  
        cv2.rectangle(output_image, (x1, y1), (x2, y2), tuple(map(int, avg_color)), -1) 

    # 保存结果图像  
    output_name = f"{output_file}clear_ocr.png"
    cv2.imwrite(output_name, output_image)  
    print(f"Output saved to: {output_name}")
    return output_name

def get_ocr_from_different_levels(xmin,ymin,screenshot_file,output,output_file):

    total_width, total_height = Image.open(screenshot_file).size
    # 将root分割成若干份子图输入
    sub_nums = [1,4,9,16]
    for j in range(len(sub_nums)):
        sub_num_ocr = sub_nums[j]
        if sub_num_ocr == 4:
            img_list, img_x_list, img_y_list = split_image_into_4(output_file, './screenshot/sub4/screenshot', xmin, ymin)
        elif sub_num_ocr == 9:
            img_list, img_x_list, img_y_list = split_image_into_9(output_file, './screenshot/sub9/screenshot9', xmin, ymin)
        elif sub_num_ocr == 16:
            img_list, img_x_list, img_y_list = split_image_into_16(output_file, './screenshot/sub16/screenshot16', xmin, ymin)
        elif sub_num_ocr == 25:
            img_list, img_x_list, img_y_list = split_image_into_25(output_file, './screenshot/sub25/screenshot25', xmin, ymin)
        elif sub_num_ocr == 36:
            img_list, img_x_list, img_y_list = split_image_into_36(output_file, './screenshot/sub36/screenshot36', xmin, ymin)
        else:
            img_list = [output_file]
            img_x_list = [xmin]
            img_y_list = [ymin]

        text_coordinates = []
        texts = []
        padding = total_height * 0.0025  # 10 生产字符框的合适间距
        
        for i, img in enumerate(img_list):
            # 获得ocr字符框列表和文字列表
            sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
            # 将字符框的坐标都调整成在原图中的坐标
            for coordinate in sub_coordinates:
                coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
                coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
                coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
                coordinate[3] = int(min(total_height,img_y_list[i] + coordinate[3] + padding))
            # 将每个子图调整后的字符框坐标和文本合并
            sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
            text_coordinates.extend(sub_coordinates_merge)
            texts.extend(sub_text_merge)
        # 将所有子图的字符框和文本合并
        merged_text, merged_text_coordinates = merge_boxes_and_texts(texts, text_coordinates)
        ocr_name = f"{output}{sub_num_ocr}.png"
        text_draw_coordinates_boxes_on_image(screenshot_file, merged_text_coordinates, ocr_name)
        
        if j == 0:
            pre_coords = merged_text_coordinates
        else:
            pre_coords = merge_min_boxes(merged_text_coordinates,pre_coords)

        if sub_num_ocr == 4:
            coords_4 = merged_text_coordinates

    ocr_name = f"{output}{sub_nums}.png"
    text_draw_coordinates_boxes_on_image(screenshot_file, pre_coords, ocr_name)
    return pre_coords, coords_4


# 一次处理一张图片
def get_perception_infos(screenshot_file,xml_file,output_file,my_screenshot_som_file,load_ocr=False,sub_num=4,text_th=0.1,bbox_th=0.5,caption=None,icon_only=False,text_only=False,clear_ocr=True):
    os.makedirs(output_file, exist_ok=True)
    # 从原图中裁剪出root部分
    xmin,ymin = extract_root_from_image(screenshot_file,xml_file,output_file) # 根据xml生成root.png
    #xmin, ymin, xmax, ymax = detect_root_area(screenshot_root,root_root,pic) # 自行检测root并生成root.png
    
    # 读取图像并分割，获得子图在原图中的起始坐标
    total_width, total_height = Image.open(screenshot_file).size

    padding = total_height * 0.0025  # 10 生产字符框的合适间距

    filename = os.path.splitext(os.path.basename(screenshot_file))[0]
    root_file = f"{output_file}root.png"
    if load_ocr:
        with open(text_coords_save_file,'rb') as f:
            data = pickle.load(f)
            merged_text_coordinates = data['ocr']
            merged_text_coordinates_4 = data['ocr_4']
            
    else:
        merged_text_coordinates, merged_text_coordinates_4 = get_ocr_from_different_levels(xmin,ymin,screenshot_file,output_file,root_file)
        with open(text_coords_save_file,'wb') as f:
            ocr_to_save = {'ocr':merged_text_coordinates,'ocr_4':merged_text_coordinates_4}
            pickle.dump(ocr_to_save,f)

    # detection module using ocr
    img_list, img_x_list, img_y_list = split_image_into_4(root_file, './screenshot/sub4/screenshot', xmin, ymin)
    text_coordinates = []
    texts = []
    padding = total_height * 0.0025  # 10 生产字符框的合适间距   
    for i, img in enumerate(img_list):
        # 获得ocr字符框列表和文字列表
        sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
        # 将字符框的坐标都调整成在原图中的坐标
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height,img_y_list[i] + coordinate[3] + padding))
        # 将每个子图调整后的字符框坐标和文本合并
        sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
        text_coordinates.extend(sub_coordinates_merge)
        texts.extend(sub_text_merge)
    # 将所有子图的字符框和文本合并
    _, merged_text_coordinates_4 = merge_boxes_and_texts(texts, text_coordinates)
    text_draw_coordinates_boxes_on_image(screenshot_file, merged_text_coordinates_4, screenshot_som_file)
    
    if clear_ocr:
        output_file_name2 = clear_ocr_from_img(merged_text_coordinates, screenshot_file, output_file)
        # 根据xml裁剪root区域，并保存至root.png
        xmin,ymin = extract_root_from_image(output_file_name2,xml_file,output_file,True)
        root_file = f"{output_file}root_clear_ocr.png"
        # 根据detect_root_area返回的结果裁剪root区域
        # with Image.open(output_file_name2) as img:
        #     cropped_img = img.crop((xmin,ymin,xmax,ymax))
        #     root_file = f"{output_file}root_clear_ocr.png"
        #     cropped_img.save(root_file)

    if sub_num == 4:
        img_list, img_x_list, img_y_list = split_image_into_4(root_file, './screenshot/sub4/screenshot', xmin, ymin)
    elif sub_num == 9:
        img_list, img_x_list, img_y_list = split_image_into_9(root_file, './screenshot/sub9/screenshot9', xmin, ymin)
    elif sub_num == 16:
        img_list, img_x_list, img_y_list = split_image_into_16(root_file, './screenshot/sub16/screenshot16', xmin, ymin)
    elif sub_num == 25:
        img_list, img_x_list, img_y_list = split_image_into_25(root_file, './screenshot/sub25/screenshot25', xmin, ymin)
    elif sub_num == 36:
        img_list, img_x_list, img_y_list = split_image_into_36(root_file, './screenshot/sub36/screenshot36', xmin, ymin)
    else:
        img_list, img_x_list, img_y_list = [root_file], [xmin], [ymin]

    coordinates = []
    confidences = []
    labels = []
    # detection module using groundingdino
    for i, img in enumerate(img_list):
        sub_coordinates, sub_confidences, sub_labels = det(img, caption, groundingdino_model, text_threshold=text_th, box_threshold=bbox_th)

        # 将子图坐标调整到原图中的坐标
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))
        # 合并调整后的坐标
        sub_coordinates,sub_confidences,sub_labels = merge_all_icon_boxes(sub_coordinates,sub_confidences,sub_labels)
        coordinates.extend(sub_coordinates)
        confidences.extend(sub_confidences)
        labels.extend(sub_labels)
    # 合并所有子图的边界框坐标
    merged_icon_coordinates,merged_icon_confidences,merged_icon_labels = merge_all_icon_boxes(coordinates,confidences,labels)
    #draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(merged_icon_coordinates), copy.deepcopy(merged_icon_confidences), copy.deepcopy(merged_icon_labels), my_screenshot_som_file, font_path, text_th=text_th, bbox_th=bbox_th)

    # 将button和text合并
    coords,logits,labels = merge_buttons_and_texts(merged_text_coordinates_4,merged_icon_coordinates,merged_icon_confidences,merged_icon_labels,icon_only,text_only)
    coords,logits,labels = merge_all_icon_boxes(coords,logits,labels)
    draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(coords), copy.deepcopy(logits), copy.deepcopy(labels), my_screenshot_som_file, font_path, text_th=text_th, bbox_th=bbox_th, with_text=True)
    
    return coords, logits, labels

def gen_csv(merged_icon_coordinates, merged_icon_confidences, merged_icon_labels, csv_filename, th):
    num_bbox = len(merged_icon_coordinates)
    filename_ext = f"{pic}.png"

    set_clickable = {'square','rectangular','input box','button','arrow','link text','icons','penguin','cross','blue icons','blue','toolbar icons','gray area',"clickable area",'red button','blue button','rectangular'}
    set_selectable = {'circle buttons','checkbox'}
    set_scrollable = {'scrollbar','triangle','combo box'}
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(num_bbox):
            logit = merged_icon_confidences[i].item()
            if logit < th:
                continue
            label = merged_icon_labels[i]
            if label in set_clickable:
                mylabel = 'clickable'
            elif label in set_selectable:
                mylabel = 'selectable'
            elif label in set_scrollable:
                mylabel = 'scrollable'
            else:
                mylabel = 'disabled'
            row = [
                filename_ext, 
                ' '.join(map(str, merged_icon_coordinates[i])),
                format(merged_icon_confidences[i].item(),'.9f'),
                mylabel
                ]
            writer.writerow(row)
    print("successfully written to csv")

def process_directory(input_dir, output_dir, csv_output_dir):
    """处理指定目录下的所有图片文件
    
    Args:
        input_dir: 输入目录，包含图片和对应的xml文件
        output_dir: 输出目录，用于保存处理结果
        csv_output_dir: CSV文件输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # 获取所有png文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    for image_file in image_files:
        # 构建完整的文件路径
        base_name = os.path.splitext(image_file)[0]
        screenshot_file = os.path.join(input_dir, image_file)
        xml_file = os.path.join(input_dir, f"{base_name}.xml")
        
        # 检查对应的xml文件是否存在
        if not os.path.exists(xml_file):
            print(f"警告: {xml_file} 不存在，跳过处理 {image_file}")
            continue
            
        # 创建输出子目录
        image_output_dir = os.path.join(output_dir, base_name)
        screenshot_som_dir = os.path.join(output_dir, "screenshot", base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(screenshot_som_dir, exist_ok=True)
        
        # 设置输出文件路径
        my_screenshot_som_file = os.path.join(screenshot_som_dir, f"{sub_num}_{caption}")
        if clear_ocr:
            my_screenshot_som_file += "_clear_ocr"
        else:
            my_screenshot_som_file += "_with_ocr"
        if icon_only:
            my_screenshot_som_file += "_icon"
        if text_only:
            my_screenshot_som_file += "_text"
            
        # 处理单个图片
        try:
            icon_coordinates, icon_confidences, icon_labels = get_perception_infos(
                screenshot_file,
                xml_file,
                image_output_dir + "/",  # get_perception_infos需要路径末尾有/
                my_screenshot_som_file,
                load_ocr=load_ocr,
                sub_num=sub_num,
                bbox_th=bbox_th,
                caption=caption,
                icon_only=icon_only,
                text_only=text_only,
                clear_ocr=clear_ocr
            )
            
            # 生成CSV文件
            csv_filename = os.path.join(csv_output_dir, f"{base_name}.csv")
            gen_csv(icon_coordinates, icon_confidences, icon_labels, csv_filename)
            
            print(f"成功处理: {image_file}")
            
        except Exception as e:
            print(f"处理 {image_file} 时出错: {str(e)}")

if __name__ == "__main__":
    # 配置参数
    mode = 'baseline'
    if mode == 'baseline':
        caption = "icons"
        sub_num = 4
        bbox_th = 0.1
        relax = False
        clear_ocr = False
        load_ocr = False
        icon_only = True
        text_only = False
    
    # 设置输入输出路径
    input_directory = "v1/jiguang"  # 包含图片和xml的目录
    output_directory = "output/v1/jiguang"  # 处理结果输出目录
    csv_output_directory = "output/v1/jiguang/csv"  # CSV文件输出目录
    
    # 处理整个目录
    process_directory(input_directory, output_directory, csv_output_directory)
