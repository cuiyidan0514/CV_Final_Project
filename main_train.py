import os  
import xml.etree.ElementTree as ET  
import xml.dom.minidom as minidom  
import cv2  
import numpy as np  
from typing import List, Dict, Tuple  
import matplotlib.pyplot as plt
import argparse
import torch
import csv
from torch.utils.data import DataLoader

from demo import *
from models import *
from dataset_loader import UIElementDataset
import csv
from dataset.eval.test_py.count.count import *
from dataset.eval.test_py.read_datas.read import *
from dataset.eval.test_py.visualization.visualization import drawing
from dataset.eval.test_py.main import run

def get_args_parser():
    parser = argparse.ArgumentParser(
        "training and testing for UI parser",add_help=False
    )
    # 基础参数
    parser.add_argument("--batch_size",default=16,type=int)
    parser.add_argument("--device",default='cpu',type=str)

    # 数据集相关（应该是需要我们自己来划分训练集和测试集）
    parser.add_argument("--image_dir",default=None,type=str)
    parser.add_argument("--xml_dir",default=None,type=str)
    parser.add_argument("--val_image_dir",default=None,type=str)
    parser.add_argument("--val_xml_dir",default=None,type=str)

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


def main(args):

    device = torch.device(args.device)

    dataset_train = UIElementDataset(image_dir=args.image_dir, xml_dir=args.xml_dir)
    dataset_val = UIElementDataset(image_dir=args.val_image_dir, xml_dir=args.val_xml_dir)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = models['object_detector'](num_classes=4)
    model.to(device)

    if args.eval:
        # 加载训练好的模型参数
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_state_dict = checkpoint['model_state_dict']
        msg = model.load_state_dict(model_state_dict, strict=False)
        print(msg)
        # 评估模型
        test_stats = run(args.iou, args.errs, args.img_root, args.xml_root, args.ott_img_root, args.csv_path, args.class_name)
        print(f"Accuracy of the object detection model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for sample in data_loader_train:
            images, targets = sample[0]
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images) # 模型输出目标检测模型获得的element_list,每个元素都包含了组件的标定框坐标和属性
            
            loss = detection_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 打印训练信息  
        print(f"Epoch {epoch+1}/{args.epochs}")  
        print(f"Train Loss: {train_loss/len(data_loader_train)}")   

        # save checkpoints
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            print("Saving model at epoch:", epoch)
            torch.save(model.state_dict(),args.model_save_path)
        
