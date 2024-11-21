import torch
import torch.nn as nn
import torchvision.models as models

'''
先让gpt写了一个简单的目标检测模型来占位
'''

class ObjectDetector(nn.Module):
    '''
    input: 
        images: PC screenshot 
        configs (Dict, optional): 配置参数
    output:
        Dict: 包含布局信息的字典  
    '''
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__(self, num_classes=4, pretrained=True)
        
        backbone = models.resnet18(pretrained=pretrained)   
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  
        self.feature_map = nn.Sequential(  
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),  
            nn.ReLU(inplace=True)  
        )  
        # 组件边界框回归头  
        self.bbox_head = nn.Sequential(  
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),  
            nn.Linear(128, 4)  # [x_min, y_min, x_max, y_max]  
        )  
        # 组件属性分类头,按助教说法应该是4个分类头:clickable,focusable,scrollable,non-interactable  
        self.cls_head = nn.Sequential(  
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),  
            nn.Linear(128, num_classes)  
        )  
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  
            nn.init.xavier_uniform_(m.weight)  
            if m.bias is not None:  
                nn.init.constant_(m.bias, 0)  
        elif isinstance(m, nn.Conv2d):  
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
        elif isinstance(m, nn.BatchNorm2d):  
            nn.init.constant_(m.weight, 1)  
            nn.init.constant_(m.bias, 0) 

    def forward(self, x, configs=None):
        features = self.backbone(x)
        features = self.feature_map(features)
        features = torch.mean(features, dim=[2,3])
        bbox_pred = self.bbox_head(features)
        cls_pred = self.cls_head(features)
        return {
            'bbox':bbox_pred,
            'cls':cls_pred
        }


# refer to https://huggingface.co/docs/transformers/model_doc/grounding-dino
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINO(nn.Module):
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def forward(self, x, configs=None):
        inputs = self.processor(images=x, text=configs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[x.size[::-1]]
        )
        return results



def process_detections(outputs, conf_threshold=0.5,iou_threshold=0.5):
    """  
    对模型预测的bbox和attribute再做一个后处理 
    Args:  
        outputs: 模型输出  
        conf_threshold: 置信度阈值  
        iou_threshold: IoU阈值  
    Returns:  
        处理后的outputs  
    """  
    bboxes = outputs['bbox']  
    cls_scores = outputs['cls']  
    
    probs = torch.softmax(cls_scores, dim=1)   
    max_probs, labels = probs.max(dim=1)  
    
    valid_detections = []  
    
    for i in range(bboxes.size(0)):  
        if max_probs[i] > conf_threshold: # 对于每个组件,只有当其对某个属性的概率预测值大于阈值时,才认为这是一个合法的组件 
            detection = {  
                'bbox': bboxes[i],  
                'label': labels[i],  
                'score': max_probs[i]  
            }  
            valid_detections.append(detection)  
    
    return valid_detections

def detection_loss(outputs, targets):
    # bbox loss
    bbox_loss = nn.functional.smooth_l1_loss(  
        outputs['bbox'],   
        targets['boxes']  
    )  
    
    # attribute loss  
    cls_loss = nn.functional.cross_entropy(  
        outputs['cls'],   
        targets['labels']  
    )  
    
    # total loss  
    total_loss = bbox_loss + cls_loss  
    
    return total_loss

models = {
    'object_detector':ObjectDetector
}
    