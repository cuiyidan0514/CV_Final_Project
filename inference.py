from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
# modelscope用于加载和使用深度学习模型
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from modelscope import snapshot_download
import concurrent.futures
import os
# PCAgent包含自定义的图像处理和目标检测功能
from PCAgent.text_localization import ocr
from PCAgent.icon_localization import det,save_training_data
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes,merge_boxes_and_texts_new, merge_buttons_and_texts, merge_all_boxes_on_logits, merge_min_boxes, merge_texts_and_icons
from PCAgent.crop import split_image_into_4,split_image_into_16,split_image_into_36,split_image_into_16_and_shift,split_image_into_25,split_image_into_9,split_image_dynamically
from PCAgent.visualize import draw_coordinates_boxes_on_image, text_draw_coordinates_boxes_on_image,visualize_gt_bbox
from detect_root_area import detect_root_area
import copy

import csv
import xml.etree.ElementTree as ET
import cv2
import pickle
from collections import Counter
from thirdparty.GroundingDINO.groundingdino.util.inference import load_model
class DetectionProcessor:
    def __init__(self):
        # 加载模型
        # self.groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
        #! revised model by xxy
        # self.groundingdino_dir = 'thirdparty/GroundingDINO'
        # self.dino_model = pipeline('grounding-dino-task', model=self.groundingdino_dir)
        self.dino_model = load_model("thirdparty/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/xiaxinyuan/.cache/modelscope/hub/AI-ModelScope/GroundingDINO/groundingdino_swint_ogc.pth")
        # self.ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
        # self.ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
        self.font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.temp_file = "temp"

    def extract_root_from_image(self, screenshot_file, xml_file, output_file, clear_ocr=False):
        if clear_ocr:
            output_file_name = f"{self.output_file}root_clear_ocr.png"
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
    
    def clear_det_from_img(self,merged_text_coordinates, screenshot_file, output_file, merged_logits=None, th=0):  
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

    def clear_ocr_from_img(self,merged_text_coordinates, screenshot_file, output_file):  
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
    
    def get_ocr_from_different_levels(self,xmin,ymin,screenshot_file,output,output_file):
        total_width, total_height = Image.open(screenshot_file).size
        # 将root分割成若干份子图输入
        
        for j in range(len(self.text_sub_num)):
            sub_num_ocr = self.text_sub_num[j]
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
                sub_text, sub_coordinates = ocr(img, self.ocr_detection, self.ocr_recognition)
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

        ocr_name = f"{output}{self.text_sub_num}.png"
        text_draw_coordinates_boxes_on_image(screenshot_file, pre_coords, ocr_name)
        return pre_coords, coords_4
    

    def get_perception_infos(self,screenshot_file,xml_file,output_file,my_screenshot_som_file,load_ocr=False,sub_num=4,text_th=0.1,bbox_th=0.5,caption=None,icon_only=False,text_only=False,clear_ocr=True):
        os.makedirs(output_file, exist_ok=True)
        # 从原图中裁剪出root部分
        xmin,ymin = self.extract_root_from_image(screenshot_file,xml_file,output_file) # 根据xml生成root.png
        #xmin, ymin, xmax, ymax = detect_root_area(screenshot_root,root_root,pic) # 自行检测root并生成root.png
        
        # 读取图像并分割，获得子图在原图中的起始坐标
        
        total_width, total_height = Image.open(screenshot_file).size

        padding = total_height * 0.0025  # 10 生产字符框的合适间距

        filename = os.path.splitext(os.path.basename(screenshot_file))[0]
        root_file = f"{output_file}root.png"
        root_width, root_height = Image.open(root_file).size
        split_x = 3- int(total_width/root_width)
        split_y = 3- int(total_height/root_height)
        img_list, img_x_list, img_y_list = split_image_dynamically(root_file, './screenshot/sub4/screenshot', xmin, ymin,split_y,split_x)

        # if sub_num == 4:
        #     img_list, img_x_list, img_y_list = split_image_into_4(root_file, './screenshot/sub4/screenshot', xmin, ymin,self.mini)
        # elif sub_num == 9:
        #     img_list, img_x_list, img_y_list = split_image_into_9(root_file, './screenshot/sub9/screenshot9', xmin, ymin)
        # elif sub_num == 16:
        #     img_list, img_x_list, img_y_list = split_image_into_16(root_file, './screenshot/sub16/screenshot16', xmin, ymin)
        # elif sub_num == 25:
        #     img_list, img_x_list, img_y_list = split_image_into_25(root_file, './screenshot/sub25/screenshot25', xmin, ymin)
        # elif sub_num == 36:
        #     img_list, img_x_list, img_y_list = split_image_into_36(root_file, './screenshot/sub36/screenshot36', xmin, ymin)
        # else:
        #     img_list, img_x_list, img_y_list = [root_file], [xmin], [ymin]
        # detection module using ocr
        text_coordinates = []
        texts = []
        padding = total_height * 0.0025  # 10 生产字符框的合适间距   
        if self.ocr_enable:
            if self.load_ocr:
                with open(self.text_coords_save_file,'rb') as f:
                    data = pickle.load(f)
                    merged_text_coordinates = data['ocr']
                    merged_text_coordinates_4 = data['ocr_4']
            else:
                for i, img in enumerate(img_list):
                    # 获得ocr字符框列表和文字列表
                    sub_text, sub_coordinates = ocr(img, self.ocr_detection, self.ocr_recognition)
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
                merged_text_coordinates, merged_text_coordinates_4 = merge_boxes_and_texts(texts, text_coordinates)
                text_draw_coordinates_boxes_on_image(screenshot_file, merged_text_coordinates_4, self.my_screenshot_som_file)
                
            if clear_ocr:
                output_file_name2 = self.clear_ocr_from_img(merged_text_coordinates, screenshot_file, output_file)
                # 根据xml裁剪root区域，并保存至root.png
                xmin,ymin = self.extract_root_from_image(output_file_name2,xml_file,output_file,True)
                root_file = f"{output_file}root_clear_ocr.png"
                # 根据detect_root_area返回的结果裁剪root区域
                # with Image.open(output_file_name2) as img:
                #     cropped_img = img.crop((xmin,ymin,xmax,ymax))
                #     root_file = f"{output_file}root_clear_ocr.png"
                #     cropped_img.save(root_file)



        # detection module using groundingdino
        coordinates = []
        confidences = []
        labels = []
        if self.get_features:
            features = []
        for i, img in enumerate(img_list):
            sub_coordinates, sub_confidences, sub_labels, sub_features= det(img, caption, self.dino_model, text_threshold=text_th, box_threshold=bbox_th)
            # 将子图坐标调整到原图中的坐标
            for coordinate in sub_coordinates:
                coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
                coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
                coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
                coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))
            # 合并调整后的坐标
            if self.get_features:
                result = merge_all_icon_boxes(bboxes=sub_coordinates,confidences=sub_confidences,labels=sub_labels,features=sub_features)
            else:
                result = merge_all_icon_boxes(bboxes=sub_coordinates,confidences=sub_confidences,labels=sub_labels)
            sub_coordinates = result['bboxes']
            sub_confidences = result['confidences']
            sub_labels = result['labels']
            if self.get_features:
                sub_features = result['features']
                features.extend(sub_features)
            coordinates.extend(sub_coordinates)
            confidences.extend(sub_confidences)
            labels.extend(sub_labels)

        # 合并所有子图的边界框坐标
        #! 优化为只合并有靠近边界的框？
        if self.get_features:
            result = merge_all_icon_boxes(bboxes=coordinates,confidences=confidences,labels=labels,features=features)
        else:
            result = merge_all_icon_boxes(bboxes=coordinates,confidences=confidences,labels=labels)
        merged_icon_coordinates = result['bboxes']
        merged_icon_confidences = result['confidences']
        merged_icon_labels = result['labels']
        if self.get_features:
            merged_icon_features = result['features']
            save_training_data(xml_file, self.train_data_save_file, merged_icon_features, merged_icon_coordinates,threshold=0.2)

        #draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(merged_icon_coordinates), copy.deepcopy(merged_icon_confidences), copy.deepcopy(merged_icon_labels), my_screenshot_som_file, font_path, text_th=text_th, bbox_th=bbox_th)

        # 将button和text合并
        if self.ocr_enable:
            coords,logits,labels = merge_buttons_and_texts(merged_text_coordinates_4,merged_icon_coordinates,merged_icon_confidences,merged_icon_labels,icon_only,text_only)
            result = merge_all_icon_boxes(bboxes=coords,confidences=logits,labels=labels)
            coords = result['bboxes']
            logits = result['confidences']
            labels = result['labels']
            draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(coords), copy.deepcopy(logits), copy.deepcopy(labels), my_screenshot_som_file, self.font_path, text_th=text_th, bbox_th=bbox_th, with_text=True)
        else:
            coords = merged_icon_coordinates
            logits = merged_icon_confidences
            labels = merged_icon_labels
            draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(coords), copy.deepcopy(logits), copy.deepcopy(labels), my_screenshot_som_file, self.font_path, text_th=text_th, bbox_th=bbox_th, with_text=False)
            gt_path = f"{my_screenshot_som_file}_gt.png"
            visualize_gt_bbox(image_path=screenshot_file,save_path=gt_path)
        return coords, logits, labels
    

    def gen_csv(self,merged_icon_coordinates, merged_icon_confidences, merged_icon_labels, csv_filename, th):
        num_bbox = len(merged_icon_coordinates)
        filename_ext = os.path.basename(csv_filename).replace('.csv', '.png')  # 从csv文件名获取图片名


        
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(num_bbox):
                logit = merged_icon_confidences[i].item()
                if logit < th:
                    continue
                label = merged_icon_labels[i]
                if label in self.caption['set_clickable']:
                    mylabel = 'clickable'
                elif label in self.caption['set_selectable']:
                    mylabel = 'selectable'
                elif label in self.caption['set_scrollable']:
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
    
    def process_directory(self, input_dir, output_dir, csv_output_dir):
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
            if self.mini:
                self.my_screenshot_som_file = os.path.join(screenshot_som_dir, f"{self.sub_num}_mini")
            else:
                self.my_screenshot_som_file = os.path.join(screenshot_som_dir, f"{self.sub_num}_{self.caption}")
            if self.clear_ocr:
                self.my_screenshot_som_file += "_clear_ocr"
            else:
                self.my_screenshot_som_file += "_with_ocr"
            if self.icon_only:
                self.my_screenshot_som_file += "_icon"
            if self.text_only:
                self.my_screenshot_som_file += "_text"
            
            self.text_coords_save_file = os.path.join(csv_output_dir, f"{base_name}.pkl")
            self.train_data_save_file = os.path.join(csv_output_dir, f"{base_name}_train_data.pkl")
            # 处理单个图片
            # try:
            icon_coordinates, icon_confidences, icon_labels = self.get_perception_infos(
                screenshot_file,
                xml_file,
                image_output_dir + "/",  # get_perception_infos需要路径末尾有/
                self.my_screenshot_som_file,
                load_ocr=self.load_ocr,
                sub_num=self.sub_num,
                text_th=self.text_th,
                bbox_th=self.bbox_th,
                caption=self.all_caption,
                icon_only=self.icon_only,
                text_only=self.text_only,
                clear_ocr=self.clear_ocr
            )
            
            # 生成CSV文件
            csv_filename = os.path.join(csv_output_dir, f"{base_name}.csv")
            self.gen_csv(icon_coordinates, icon_confidences, icon_labels, csv_filename,th = self.bbox_th)
            
            print(f"成功处理: {image_file}")
                
            # except Exception as e:
            #     print(f"处理 {image_file} 时出错: {str(e)}")

    def set_params(self,mode):
        if mode == 'baseline':
            # self.caption = "icons"
            self.sub_num = 4
            self.text_sub_num = [4]
            self.bbox_th = 0.1
            self.relax = False
            self.clear_ocr = False
            self.load_ocr = True
            self.icon_only = True
            self.text_only = False
            self.mini = True
            self.ocr_enable = True
            self.get_features = False
            # set_clickable = {'square','rectangular','input box','button','arrow','link text','icons','penguin','cross','blue icons','blue','toolbar icons','gray area',"clickable area",'red button','blue button','rectangular','toolbar icon'}
            if self.mini:
                set_clickable = {'square','rectangular','inputBox','button','arrow','linkText','icons','cross',"clickableArea",'rectangular','toolbarIcon'}
            else:
                set_clickable = {'square','rectangular','input box','button','arrow','link text','icons','penguin','cross','blue icons','blue','toolbar icons','gray area',"clickable area",'red button','blue button','rectangular','toolbar icon'}
            set_selectable = {'circleButtons','checkbox'}
            set_scrollable = {'scrollbar','triangle','comboBox','slidingArea','slide bar','dropdown'}
            self.caption = {'set_clickable':set_clickable,'set_selectable':set_selectable,'set_scrollable':set_scrollable}
            # self.all_caption = ' . '.join(list(set_clickable | set_selectable | set_scrollable))
        if mode == 'my_train':
            set_clickable = {'icon','button'}
            set_selectable = {'checkbox'}
            set_scrollable = {'triangle'}
            self.caption = {'set_clickable':set_clickable,'set_selectable':set_selectable,'set_scrollable':set_scrollable}

            # self.all_caption = ' . '.join(list(set_clickable | set_selectable | set_scrollable))
            # set_clickable = {'Please detect all icons separately'}
            self.all_caption =' . '.join(set_clickable)
            self.sub_num = 9
            self.text_sub_num = [9]
            self.text_th = 0.05
            self.bbox_th = 0.05
            self.relax = False
            self.clear_ocr = False
            self.load_ocr = True
            self.icon_only = True
            self.text_only = False
            self.mini = True
            self.ocr_enable = False
            self.get_features = True


import argparse
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='图像处理参数设置')
    # parser.add_argument('--mode', type=str, default='baseline', help='处理模式')
    # parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    # parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    # parser.add_argument('--csv_output_dir', type=str, required=True, help='CSV输出目录')

    # args = parser.parse_args()  # 解析命令行参数

    # my_detection = DetectionProcessor()
    # my_detection.set_params(args.mode)  # 使用解析的模式参数
    # input_dir = args.input_dir  # 使用解析的输入目录
    # output_dir = args.output_dir  # 使用解析的输出目录
    # csv_output_dir = args.csv_output_dir  # 使用解析的CSV输出目录
    # my_detection.process_directory(input_dir, output_dir, csv_output_dir)

    #! for debug, sometimes set load_ocr=True
    my_detection = DetectionProcessor()
    my_detection.set_params('my_train')
    input_dir = 'v1/skype'
    output_dir = 'output/v1/skype'
    csv_output_dir = 'output/v1/skype/csv'
    my_detection.process_directory(input_dir, output_dir, csv_output_dir)


