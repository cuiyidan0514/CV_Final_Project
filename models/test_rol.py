import torch
import os
import csv
from PIL import Image
import torchvision.transforms as transforms
from rol import UIElementDetector
import pandas as pd

def load_model(model_path, device):
    """加载训练好的模型"""
    model = UIElementDetector(num_classes=4)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def process_image(image_path, transform):
    """处理图像"""
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def predict_and_save(model, csv_input_path, csv_output_path, image_root_dir, device):
    """预测并保存结果"""
    transform = transforms.ToTensor()
    
    # 读取输入CSV，指定分隔符为逗号
    df = pd.read_csv(csv_input_path, header=None, 
                     names=['image_name', 'bbox', 'confidence', 'label'],
                     sep=',')
    
    # 创建输出CSV，确保使用逗号分隔符，且不使用引号包围字段
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        # 按图像分组处理
        for image_name, group in df.groupby('image_name'):
            image_path = os.path.join(image_root_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found")
                continue
                
            # 处理图像
            image = process_image(image_path, transform)
            image = image.unsqueeze(0).to(device)
            
            for _, row in group.iterrows():
                # 处理bbox
                bbox_str = row['bbox']
                bbox = [int(x) for x in bbox_str.split()]
                bbox_tensor = torch.tensor([bbox], dtype=torch.float32).to(device)
                
                # 模型预测
                with torch.no_grad():
                    outputs = model(image, bbox_tensor)
                    confidence, predicted = torch.max(outputs, 1)
                    confidence = torch.softmax(outputs, dim=1)[0][predicted]
                
                # 将预测结果转换为标签
                label_map = {
                    0: 'clickable',
                    1: 'selectable',
                    2: 'scrollable',
                    3: 'disabled'
                }
                predicted_label = label_map[predicted.item()]
                
                # 写入CSV，确保格式完全匹配
                row_data = f"{image_name},{bbox_str},{confidence.item():.9f},{predicted_label}"
                csvfile.write(row_data + '\n')
            
            print(f"Processed {image_name}")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置路径
    model_path = 'checkpoint_20241219_150420_acc_0.8742.pth'
    csv_input_path = 'eval/jiguang.csv'
    csv_output_path = 'eval/jiguang_rol_0.87.csv'
    image_root_dir = 'v1/jiguang'  # 请替换为实际的图像目录路径
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 预测并保存结果
    predict_and_save(model, csv_input_path, csv_output_path, image_root_dir, device)
    
    print("Testing completed. Results saved to", csv_output_path)

if __name__ == '__main__':
    main()