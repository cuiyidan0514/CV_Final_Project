from PIL import Image, ImageDraw
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download
import concurrent.futures
import os

from PCAgent.text_localization import ocr
from PCAgent.icon_localization import det
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes,merge_boxes_and_texts_new
from run_original import split_image_into_4,draw_coordinates_boxes_on_image
import copy

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
        """合并所有部分的结果"""
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
        """处理完整图片的主函数"""
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



groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

# test input image
screenshot_file = "v1/onedrive/frame_1.png"
# path for saving the output image
screenshot_som_file = "./screenshot/screenshot_som.png"
# draw text box or not
DRAW_TEXT_BOX = 1
USE_SOM = 1
ICON_CAPTION = 1
LOCATION_INFO = 'center'
caption_call_method = "local"
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" #! for linux

temp_file = "temp"


def get_perception_infos(screenshot_file):
    total_width, total_height = Image.open(screenshot_file).size
    split_image_into_4(screenshot_file, './screenshot/screenshot')
    img_list = ['./screenshot/screenshot_part_1.png', './screenshot/screenshot_part_2.png',
                './screenshot/screenshot_part_3.png', './screenshot/screenshot_part_4.png']
    img_x_list = [0, total_width/2, 0, total_width/2]
    img_y_list = [0, 0, total_height/2, total_height/2]
    coordinates = []
    texts = []
    padding = total_height * 0.0025  # 10
    # OCR module using resnet18 and convnextTiny
    for i, img in enumerate(img_list):
        width, height = Image.open(img).size

        sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height,img_y_list[i] + coordinate[3] + padding))

        sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
        coordinates.extend(sub_coordinates_merge)
        texts.extend(sub_text_merge)
    merged_text, merged_text_coordinates = merge_boxes_and_texts(texts, coordinates)

    coordinates = []
    # detection module using groundingdino
    for i, img in enumerate(img_list):
        width, height = Image.open(img).size
        sub_coordinates = det(img, "icon", groundingdino_model)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))

        sub_coordinates = merge_all_icon_boxes(sub_coordinates)
        coordinates.extend(sub_coordinates)
    merged_icon_coordinates = merge_all_icon_boxes(coordinates)

    if DRAW_TEXT_BOX == 1:
        rec_list = merged_text_coordinates + merged_icon_coordinates
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(rec_list), screenshot_som_file, font_path)
    else:
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(merged_icon_coordinates), screenshot_som_file, font_path)
    

    return total_width, total_height

get_perception_infos(screenshot_file)