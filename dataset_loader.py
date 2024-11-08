
import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset 
from typing import List, Dict, Tuple  
import xml.etree.ElementTree as ET  
from PIL import Image  
import torchvision.transforms as transforms  

'''
给定图像和xml文件,将图像和xml都转换为方便后续训练的格式
'''

class UIElementDataset(Dataset):  

    def __init__(self,   
                 image_dir: str,   
                 xml_dir: str,   
                 transform=None,
                 target_transform=None):  
        """  
        Args:  
            image_dir: 图像文件夹路径  
            xml_dir: XML标注文件夹路径  
            transform: 图像变换  
        """  
        self.image_dir = image_dir  
        self.xml_dir = xml_dir  
        if transform is None:  
            self.transform = transforms.Compose([  
                transforms.Resize((224, 224)),  
                transforms.ToTensor(),  
                transforms.Normalize(  
                    mean=[0.485, 0.456, 0.406],   
                    std=[0.229, 0.224, 0.225]  
                )  
            ])  
        else:  
            self.transform = transform
        self.target_transform = target_transform
        
        # 获取所有图像文件  
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  
        self.class_map = self._build_class_map()

    def _build_class_map(self):  
        """  
        构建类别映射  
        """  
        class_set = set()  
        for img_name in self.images:  
            xml_path = os.path.join(  
                self.xml_dir,   
                os.path.splitext(img_name)[0] + '.xml'  
            )  
            
            # 解析XML  
            tree = ET.parse(xml_path)  
            root = tree.getroot()  
            
            # 收集所有类别  
            for obj in root.findall('object'):  
                class_name = obj.find('name').text  
                class_set.add(class_name)  
        
        # 创建类别到索引的映射  
        return {  
            cls: idx   
            for idx, cls in enumerate(sorted(class_set))  
        }
    
    def _parse_xml(self, xml_path):  
        """  
        解析XML文件获取边界框和标签  
        """  
        tree = ET.parse(xml_path)  
        root = tree.getroot()

        # 获取图像原始尺寸  
        size = root.find('size')  
        orig_width = int(size.find('width').text)  
        orig_height = int(size.find('height').text)  
        
        boxes = []  
        labels = []  
        
        # 对xml里的每个组件的gt进行解析
        for obj in root.findall('object'):  
            # 类别名称  
            class_name = obj.find('name').text
            # 获取边界框  
            bbox = obj.find('bndbox')  
            xmin = int(bbox.find('xmin').text)  
            ymin = int(bbox.find('ymin').text)  
            xmax = int(bbox.find('xmax').text)  
            ymax = int(bbox.find('ymax').text) 
            # 标准化边界框  
            boxes.append([  
                xmin / orig_width,  
                ymin / orig_height,  
                xmax / orig_width,  
                ymax / orig_height  
            ]) 
            # 获取标签  
            labels.append(self.class_map[class_name])

        boxes = torch.tensor(boxes, dtype=torch.float32)  
        labels = torch.tensor(labels, dtype=torch.long)  

        target = {  
            'boxes': boxes,  
            'labels': labels,  
            'orig_size': (orig_width, orig_height)  
        }
        
        return target 
    
    def __len__(self):  
        return len(self.images)  
    
    def __getitem__(self, idx):  
        # 加载图像  
        img_name = self.images[idx]  
        img_path = os.path.join(self.image_dir, img_name)  
        image = Image.open(img_path).convert('RGB')  
        
        # 加载对应的XML标注  
        xml_name = os.path.splitext(img_name)[0] + '.xml'  
        xml_path = os.path.join(self.xml_dir, xml_name)  
        
        # 图像变换  
        if self.transform:
            image = self.transform(image)

        # 解析XML
        target = self._parse_xml(self,xml_path)  
        
        return image, target
    

    def _get_label_id(self, label):  
        """  
        将标签映射到ID  
        """  
        label_map = {  
            'clickable': 0,  
            # 根据需要添加其他标签  
        }  
        return label_map.get(label, -1)  
