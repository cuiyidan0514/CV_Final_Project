
import os  
import torch  
from torch.utils.data import Dataset 
import xml.etree.ElementTree as ET  
from PIL import Image  
import torchvision.transforms as transforms  

'''
给定图像和xml文件,将图像和xml都转换为方便后续训练的格式
'''

class UIElementDataset(Dataset):  

    def __init__(self, base_dir: str, transform=None):  
        """  
        Args:  
            base_dir: 文件夹路径   
            transform: 图像变换  
        """  
        
        self.base_dir = base_dir  

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
        
        # 获取所有图像文件  
        self.data = []
        for folder in os.listdir(base_dir):
            # 遍历每个子文件夹
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    # 遍历子文件夹下的所有图像
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path,img_name)
                        xml_path = os.path.splitext(img_path)[0] + '.xml'
                        if os.path.exists(xml_path):
                            self.data.append((img_path, xml_path))

    
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
        orig_depth = int(size.find('depth').text) 
        
        boxes = []  
        labels = []  
        
        # 对xml里的每个组件的gt进行解析
        for obj in root.findall('object'):  
            # 类别名称  
            class_name = obj.find('name').text
            # 获取标签,用0-3来表示4个类别
            label = self._get_label_id(class_name)
            if label == 4 or label == -1:
                continue
            labels.append(label)
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

        boxes = torch.tensor(boxes, dtype=torch.float32)  
        labels = torch.tensor(labels, dtype=torch.int32)  

        target = {  
            'boxes': boxes,  
            'labels': labels,  
            'orig_size': (orig_width, orig_height,orig_depth)  
        }
        
        return target 
    
    def __len__(self):  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        # 加载图像  
        img_path, xml_path = self.data[idx]  
        image = Image.open(img_path).convert('RGB')  
        
        # 图像变换  
        if self.transform:
            image = self.transform(image)

        # 解析XML,返回：bbox，label，orig_size
        target = self._parse_xml(xml_path)  
        
        return image, target
    

    def _get_label_id(self, label):  
        """  
        将标签映射到ID  
        """  
        label_map = {  
            'clickable': 0,
            'Clickable': 0,
            'level_0_clickable':0,
            'level_1_clickable':0,
            'selectable': 1,
            'Selectable': 1,
            'level_0_selectable':1,
            'level_1_selectable':1,
            'scrollable': 2,
            'Scrollable': 2,
            'level_0_scrollable':2,
            'level_1_scrollable':2,
            'disabled': 3,
            'Disabled': 3,
            'level_0_disabled':3,
            'level_1_disabled':3,
            'level_0':4,
            'level_1':4,
            'level_2':4,
            'root':4,
            'Root':4,
        }  
        return label_map.get(label, -1)  
