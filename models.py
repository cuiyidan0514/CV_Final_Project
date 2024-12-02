import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

'''
先让gpt写了一个简单的目标检测模型来占位
用预训练好的resnet18作为backbone，增添一个特征处理层，一个bbox回归头，一个bbox置信度预测头，一个label分类头
三个预测头得到的bbox，confidence，label即为模型输出
'''

class ObjectDetector(nn.Module):
    '''
    input: 
        num_classes: 标签的数量，默认为5，4个组件以及1个噪声 
        num_bboxes: 每张图像预测的组件的数量，假设为10
    output:
        bbox_pred, confidence, label_pred：预测的bbox,对bbox的置信度，和对应的label 
    '''
    def __init__(self, num_classes=5, num_bboxes=10, pretrained=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_boxes = num_bboxes
        
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)   
        self.backbone = nn.Sequential(*list(backbone.children())[:-2]) #去掉最后的全连接层和分类层  
        self.feature_map = nn.Sequential(  
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),  
            nn.ReLU(inplace=True)  
        )  
        # 组件边界框回归头,每个bbox都需要两个点的坐标，即[x_min, y_min, x_max, y_max]   
        self.bbox_head = nn.Sequential(  
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),  
            nn.Linear(128, num_bboxes * 4) 
        )
        # 置信度预测头,表明对每个bbox的置信度  
        self.confidence_head = nn.Sequential(  
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),  
            nn.Linear(128, num_bboxes),  
            nn.Sigmoid()
        )  
        # 组件属性分类头,按助教说法应该是4个分类头:clickable,selectable,scrollable,disabled  
        self.cls_head = nn.Sequential(  
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),  
            nn.Linear(128, num_bboxes * num_classes)  
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

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_map(features)
        features = torch.mean(features, dim=[2,3])
        # 预测bbox
        bbox_pred = self.bbox_head(features)
        bbox_pred = bbox_pred.view(-1, self.num_boxes, 4)
        # 预测label
        label_pred = self.cls_head(features)
        label_pred = label_pred.view(-1, self.num_boxes, self.num_classes)
        label_pred = torch.softmax(label_pred, dim=2)
        max_probs, max_labels = label_pred.max(dim=2)
        # bbox的置信度
        confidence = self.confidence_head(features)  
        confidence = confidence.view(-1, self.num_boxes)  
        
        return bbox_pred, confidence, max_labels


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
            target_sizes=[x.shape[:2]]
        )[0]
                
        # 初始化输出张量
        batch_size = x.shape[0]
        bbox_pred = torch.zeros((batch_size, self.num_bboxes, 4), device=self.device)
        confidence = torch.zeros((batch_size, self.num_bboxes), device=self.device)
        max_labels = torch.zeros((batch_size, self.num_bboxes), dtype=torch.long, device=self.device)
        
        
        # 将结果转换为所需格式
        if len(results['boxes']) > 0:
            num_detected = min(len(results['boxes']), self.num_bboxes)
            bbox_pred[0, :num_detected] = results['boxes'][:num_detected]
            confidence[0, :num_detected] = results['scores'][:num_detected]
            max_labels[0, :num_detected] = results['labels'][:num_detected]
        
        bboxes,labels, scores = process_detections(bbox_pred, max_labels, confidence)
        return bboxes, labels, scores



def process_detections(bbox_pred, label_pred, conf_threshold=0.3):
    """  
    对模型预测的bbox和attribute再做一个后处理，转换为需要的格式 
    Args:  
        conf_threshold: 置信度阈值  
    Returns:  
        [图像名，预测坐标，预测坐标自信度，预测标签]  
    """   
    max_probs, max_labels = label_pred.max(dim=2)
    
    bboxes = []
    labels = []
    scores = []  
    
    # 对于预测到的所有bbox和label，只有对label的置信度高于阈值，才认为这是一个有效的预测
    for i in range(bbox_pred.shape[1]):  
        if max_probs[0][i] > conf_threshold:
            bboxes.append(bbox_pred[0][i])
            labels.append(max_labels[0][i])
            scores.append(max_probs[0][i])
    
    return bboxes, labels, scores

def detection_loss(pred_bbox, pred_label, scores, bboxs, labels):
    # bbox_loss = nn.functional.smooth_l1_loss(pred_bbox,bboxs)  
    # label_loss = nn.functional.cross_entropy(pred_label,labels)  
    # total_loss = bbox_loss + label_loss  
    # return total_loss
    pass

    