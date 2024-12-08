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
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes,merge_boxes_and_texts_new, merge_buttons_and_texts, merge_all_boxes_on_logits
from run_original import split_image_into_4,split_image_into_16,split_image_into_36,draw_coordinates_boxes_on_image,split_image_into_16_and_shift,split_image_into_25,text_draw_coordinates_boxes_on_image,split_image_into_9
import copy

import csv
import xml.etree.ElementTree as ET

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
font_path = r'C:\Windows\Fonts\Arial.ttf' # windows
temp_file = "temp"


# test input image
screenshot_file = "./dataset/train_dataset/jiguang/frame_2_1.png"
xml_file = "./dataset/train_dataset/jiguang/frame_2_1.xml"
output_file = "./dataset/train_root_dataset/jiguang/frame_2_1.png"
screenshot_som_file = "./screenshot/jiguang/frame_2_1/"
csv_filename = "./output/jiguang.csv"


# 从原图中将root区域裁剪出来
def extract_root_from_image(screenshot_file, xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == "root":
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            with Image.open(screenshot_file) as img:
                cropped_img = img.crop((xmin,ymin,xmax,ymax))
                cropped_img.save(output_file)
                print(f'Saved cropped image to {output_file}')
            
            return xmin,ymin


# 一次处理一张图片
def get_perception_infos(screenshot_file,xml_file,output_file,my_screenshot_som_file,sub_num=4,text_th=0.1,bbox_th=0.5,caption=None):
    
    # 从原图中裁剪出root部分
    xmin,ymin = extract_root_from_image(screenshot_file,xml_file,output_file)
    
    # 读取图像并分割，获得子图在原图中的起始坐标
    total_width, total_height = Image.open(screenshot_file).size
    
    # 将root分割成若干份子图输入
    if sub_num == 4:
        img_list, img_x_list, img_y_list = split_image_into_4(output_file, './screenshot/sub4/screenshot', xmin, ymin)
    elif sub_num == 9:
        img_list, img_x_list, img_y_list = split_image_into_9(output_file, './screenshot/sub9/screenshot9', xmin, ymin)
    elif sub_num == 16:
        img_list, img_x_list, img_y_list = split_image_into_16(output_file, './screenshot/sub16/screenshot16', xmin, ymin)
    elif sub_num == 25:
        img_list, img_x_list, img_y_list = split_image_into_25(output_file, './screenshot/sub25/screenshot25', xmin, ymin)
    elif sub_num == 36:
        img_list, img_x_list, img_y_list = split_image_into_36(output_file, './screenshot/sub36/screenshot36', xmin, ymin)
    else:
        img_list, img_x_list, img_y_list = split_image_into_16_and_shift(output_file, './screenshot/sub16shift/screenshot16', xmin, ymin)

    text_coordinates = []
    texts = []
    padding = total_height * 0.0025  # 10 生产字符框的合适间距
    
    #OCR module using resnet18 and convnextTiny
    for i, img in enumerate(img_list):
        width, height = Image.open(img).size
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
    #text_draw_coordinates_boxes_on_image(screenshot_file, merged_text_coordinates, my_screenshot_som_file)

    coordinates = []
    confidences = []
    labels = []
    # detection module using groundingdino
    for i, img in enumerate(img_list):
        image = Image.open(img)
        width, height = image.size
        # blur_img = image.filter(ImageFilter.GaussianBlur(radius=5))
        # scale = 1
        # enlarged_image = image.resize((int(width * scale), int(height * scale)))
        # enlarged_image.save("./screenshot/temp.png")
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
    draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(merged_icon_coordinates), copy.deepcopy(merged_icon_confidences), copy.deepcopy(merged_icon_labels), my_screenshot_som_file, font_path, text_th=text_th, bbox_th=bbox_th)

    # 将button和text合并
    coords,logits,labels = merge_buttons_and_texts(merged_text_coordinates,merged_icon_coordinates,merged_icon_confidences,merged_icon_labels)
    coords,logits,labels = merge_all_icon_boxes(coords,logits,labels)
    draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(coords), copy.deepcopy(logits), copy.deepcopy(labels), my_screenshot_som_file, font_path, text_th=text_th, bbox_th=bbox_th, with_text=True)

    return coords, merged_icon_confidences, merged_icon_labels

def gen_csv(merged_icon_coordinates, merged_icon_confidences, merged_icon_labels, csv_filename):
    num_bbox = len(merged_icon_coordinates)
    filename = os.path.splitext(os.path.basename(screenshot_file))[0]
    filename_ext = f"{filename}.png"
    with open(csv_filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(num_bbox):
            row = [
                filename_ext, 
                ' '.join(map(str, merged_icon_coordinates[i])),
                merged_icon_confidences[i].item(),
                #merged_icon_labels[i]
                'selectable'
                ]
            writer.writerow(row)
    print("successfully written to csv")



# 多提示词
# caption_list = ["circle buttons","square buttons"]
# sub_num = [4,9]
# bbox_ths = [0.35,0.15]
# for i,(caption,nn,bbox_th) in enumerate(zip(caption_list,sub_num,bbox_ths)):
#     my_screenshot_som_file = f"{screenshot_som_file}{nn}_{caption}.png"
#     # 得到的coords彼此之间都互不重叠
#     coords, logits, labels = get_perception_infos(screenshot_file,xml_file,output_file,my_screenshot_som_file,sub_num=nn,bbox_th=bbox_th,caption=caption)
#     # 将新的caption得到的bbox与已有的bbox合并
#     if  i == 0:
#         pre_coords = coords
#         pre_logits = logits
#         pre_labels = labels
#     else:
#         pre_coords, pre_logits, pre_labels = merge_all_boxes_on_logits(coords, logits, labels, pre_coords, pre_logits, pre_labels)

# pre_coords, pre_logits, pre_labels = merge_all_icon_boxes(pre_coords, pre_logits, pre_labels)
# filename = f"{screenshot_som_file}{sub_num}.png"
# draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(pre_coords), copy.deepcopy(pre_logits), copy.deepcopy(pre_labels), filename, font_path, bbox_th=bbox_ths, with_text=True)

# 单提示词
caption = "arrow buttons"
sub_num = 9
bbox_th = 0.1
my_screenshot_som_file = f"{screenshot_som_file}{sub_num}_{caption}.png"
merged_icon_coordinates, merged_icon_confidences, merged_icon_labels = get_perception_infos(screenshot_file,xml_file,output_file,my_screenshot_som_file,sub_num,bbox_th=0.15,caption=caption)

#gen_csv(merged_icon_coordinates, merged_icon_confidences, merged_icon_labels, csv_filename)