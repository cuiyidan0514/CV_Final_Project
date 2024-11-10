import os  
import xml.etree.ElementTree as ET  
import xml.dom.minidom as minidom  
import cv2  
import numpy as np  
from typing import List, Dict, Tuple  
import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import summary
from tqdm import tqdm
from datetime import datetime
import argparse


from run import *
from models import *
from dataset_loader import UIElementDataset
import csv
# from eval.test_py.count.count import *
# from eval.test_py.read_datas.read import *
# from eval.test_py.visualization.visualization import drawing
# from eval.test_py import main

def get_args_parser():
    parser = argparse.ArgumentParser(
        "training and testing for UI parser",add_help=False
    )
    # 基础参数
    parser.add_argument("--batch_size",default=1,type=int)

    # 数据集相关（应该是需要我们自己来划分训练集和测试集）
    parser.add_argument("--train_dir",default=None,type=str)
    parser.add_argument("--val_dir",default=None,type=str)

    # 训练相关
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--model_save_path", default=None, type=str)

    # 评估相关
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--iou", default=0.5, type=float)
    parser.add_argument("--errs", default=0.3, type=float)
    parser.add_argument("--img_root", default=None, type=str)
    parser.add_argument("--xml_root", default=None, type=str)
    parser.add_argument("--ott_img_root", default=None, type=str)
    parser.add_argument("--csv_path", default=None, type=str)
    parser.add_argument("--class_name", default=None, type=str)

    # 日志和检查点相关参数  
    parser.add_argument("--log_dir", default="./logs", type=str)  
    parser.add_argument("--ckpt_dir", default="./checkpoints", type=str)  
    parser.add_argument("--log_interval", default=10, type=int)  
    parser.add_argument("--save_interval", default=5, type=int)
    
    return parser

def create_logger(log_dir):  
    """创建日志目录和TensorBoard writer"""  
    os.makedirs(log_dir, exist_ok=True)  
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  
    log_path = os.path.join(log_dir, current_time)  
    writer = summary(log_path)  
    return writer  

def save_checkpoint(model, optimizer, epoch, ckpt_dir):  
    """保存模型检查点"""  
    os.makedirs(ckpt_dir, exist_ok=True)  
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")  
    torch.save({  
        'epoch': epoch,  
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
    }, ckpt_path)  
    print(f"Checkpoint saved: {ckpt_path}")  

def load_checkpoint(model, optimizer, ckpt_path):  
    """加载模型检查点"""  
    if os.path.exists(ckpt_path):  
        checkpoint = torch.load(ckpt_path)  
        model.load_state_dict(checkpoint['model_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        start_epoch = checkpoint['epoch']  
        print(f"Loaded checkpoint from epoch {start_epoch}")  
        return start_epoch  
    return 0 


def main(args):

    writer = create_logger(args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    dataset_train = UIElementDataset(base_dir=args.train_dir)
    
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    model = ObjectDetector()
    model.to(device)

    start_epoch = 0
    # 可选择恢复训练
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)

    # if args.eval:
    #     # 加载训练好的模型参数
    #     checkpoint = torch.load(args.resume, map_location="cpu")
    #     model_state_dict = checkpoint['model_state_dict']
    #     msg = model.load_state_dict(model_state_dict, strict=False)
    #     print(msg)
    #     # 评估模型
    #     test_stats = evaluate(args.iou, args.errs, args.img_root, args.xml_root, args.ott_img_root, args.csv_path, args.class_name)
    #     print(f"Accuracy of the object detection model: {test_stats['acc1']:.1f}%")
    #     exit(0)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        # 使用tqdm添加进度条  
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{args.epochs}")  

        for batch_idx,sample in enumerate(pbar):
            image,target = sample
            image = image.to(device) #[batch_size,3,224,224] 图像形状统一处理为3*224*224
            
            bboxes = target['boxes'].to(device) #[batch_size, bboxes_num, 4]
            labels = target['labels'].to(device) #[batch_size, labels_num]
            assert(labels.shape[1] == bboxes.shape[1])

            optimizer.zero_grad()

            pred_bbox, pred_confidence, pred_label = model(image)
            '''
            pred_bbox:[batch_size,num_bbox,4]
            pred_confidence:[batch_size,num_bbox]
            pred_label:[batch_size,num_bbox]
            '''

            # 这个得根据助教的eval来改
            loss = detection_loss(pred_bbox, pred_confidence, pred_label, bboxes, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 日志记录  
            if batch_idx % args.log_interval == 0:  
                writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader_train) + batch_idx)  
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})  

        # 平均训练损失  
        avg_loss = train_loss / len(data_loader_train)  
        writer.add_scalar('Epoch Average Loss', avg_loss, epoch)  

        # save checkpoints
        if (epoch + 1) % args.save_interval == 0:  
            save_checkpoint(model, optimizer, epoch + 1, args.ckpt_dir) 

    writer.close() 


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
        
