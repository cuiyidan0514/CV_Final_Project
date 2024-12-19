import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import os

class UIElementDataset(Dataset):
    def __init__(self, root_dir):
        """
        参数:
            root_dir: 数据集根目录
        """
        self.root_dir = root_dir

        self.transform = transforms.ToTensor()  # 仅做基础的转换
        
        # 获取所有png文件
        
        self.image_files = []
        
        # 遍历所有子文件夹
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):  # 确保是文件夹
                # 获取该子文件夹下的所有png文件
                png_files = [os.path.join(subdir_path, f) 
                            for f in os.listdir(subdir_path)
                            if f.endswith('.png') and 'annotated' not in f]
                self.image_files.extend(png_files)
        
    def __len__(self):
        return len(self.image_files)
    
    def _get_label_id(self, label):
        label_map = {  
            'clickable': 0, 'Clickable': 0,
            'level_0_clickable': 0, 'level_1_clickable': 0,
            'selectable': 1, 'Selectable': 1,
            'level_0_selectable': 1, 'level_1_selectable': 1,
            'scrollable': 2, 'Scrollable': 2,
            'level_0_scrollable': 2, 'level_1_scrollable': 2,
            'disabled': 3, 'Disabled': 3,
            'level_0_disabled': 3, 'level_1_disabled': 3
        }
        return label_map.get(label, -1)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)  # 转换为tensor [C, H, W]
        
        # 读取对应的XML文件
        xml_path = img_path.replace('.png', '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        # 获取所有目标
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self._get_label_id(name)
            
            # 只处理0-3的标签
            if label not in [0, 1, 2, 3]:
                continue
                
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            
        # 转换为tensor
        if len(boxes) == 0:
            return None
            
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class UIElementDetector(nn.Module):
    def __init__(self, num_classes=4):
        super(UIElementDetector, self).__init__()
        
        # 特征提取
        # 特征提取 - 减少通道数从64,128变为32,64
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        
        # ROI Pooling层 - 从7x7减小到5x5
        self.roi_pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # 分类器 - 减少中间层神经元数量
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),  # 64通道 * 5 * 5
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, boxes):
        # 提取整张图的特征
        features = self.features(image) # [batch_size, 128, H, W]
        
        roi_features = []
        # 对每个box进行ROI Pooling
        for box in boxes:
            # xmin, ymin, xmax, ymax = box
            xmin = box[0].long()
            ymin = box[1].long()
            xmax = box[2].long()
            ymax = box[3].long()
            roi = features[:, :, ymin:ymax, xmin:xmax]
            pooled_roi = self.roi_pool(roi)  # [1, 128, 7, 7]
            roi_features.append(pooled_roi)
        
        # 将所有ROI特征拼接
        roi_features = torch.cat(roi_features, dim=0)  # [num_rois, 128, 7, 7]
        
        # 展平
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        # 分类
        outputs = self.classifier(roi_features)
        
        return outputs


