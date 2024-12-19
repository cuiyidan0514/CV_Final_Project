import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from torch.utils.data import random_split
from rol import UIElementDataset, UIElementDetector
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import time

def create_data_loaders(root_dir, batch_size=32):
    """
    创建训练集和测试集的数据加载器
    """
    # 创建完整数据集
    full_dataset = UIElementDataset(root_dir)
    
    # 计算划分长度
    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)
    test_size = total_size - train_size
    
    # 随机划分数据集
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: [item for item in x if item is not None]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: [item for item in x if item is not None]
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, num_epochs=50, device='cuda', resume=True, checkpoint_path='best_model.pth'):
    """
    训练模型并记录训练过程，支持断点续训
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建tensorboard writer
    writer = SummaryWriter('runs/ui_element_detector')
    
    # 加载检查点（如果存在）
    start_epoch = 0
    best_accuracy = 0.0
    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            print(f"Resuming from epoch {start_epoch} with best accuracy: {best_accuracy:.4f}")
        else:
            print("Checkpoint does not contain model_state_dict or optimizer_state_dict. Starting from scratch.")
        
    # 记录模型结构
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    dummy_boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32).to(device)
    writer.add_graph(model, (dummy_input, dummy_boxes))
    
    global_step = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        epoch_predictions = []
        epoch_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            for item in batch:
                images = item['image'].to(device).unsqueeze(0)
                boxes = item['boxes'].to(device)
                labels = item['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images, boxes)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # 记录训练信息
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                epoch_predictions.extend(predictions.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
                
                # 每100步记录一次详细信息
                if batch_idx % 100 == 99:
                    writer.add_scalar('Training/BatchLoss', 
                                    loss.item(), 
                                    global_step)
                    global_step += 1
        
        # 计算epoch级别的指标
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_score(epoch_labels, epoch_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            epoch_labels, 
            epoch_predictions, 
            average='weighted'
        )
        
        # 记录epoch级别的指标
        writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
        writer.add_scalar('Training/Accuracy', epoch_accuracy, epoch)
        writer.add_scalar('Training/Precision', precision, epoch)
        writer.add_scalar('Training/Recall', recall, epoch)
        writer.add_scalar('Training/F1', f1, epoch)
        
        # 评估模型
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        writer.add_scalar('Testing/Loss', test_metrics['loss'], epoch)
        writer.add_scalar('Testing/Accuracy', test_metrics['accuracy'], epoch)
        writer.add_scalar('Testing/Precision', test_metrics['precision'], epoch)
        writer.add_scalar('Testing/Recall', test_metrics['recall'], epoch)
        writer.add_scalar('Testing/F1', test_metrics['f1'], epoch)
        
        # 保存检查点，包含更多信息
        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'test_metrics': test_metrics
            }
            torch.save(checkpoint, checkpoint_path)
            # 同时保存一个带时间戳的版本
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            backup_path = f'checkpoint_{timestamp}_acc_{best_accuracy:.4f}.pth'
            torch.save(checkpoint, backup_path)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        print(f'Test Loss: {test_metrics["loss"]:.4f}, '
              f'Accuracy: {test_metrics["accuracy"]:.4f}')
    
    writer.close()

def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型性能
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            for item in batch:
                images = item['image'].to(device).unsqueeze(0)
                boxes = item['boxes'].to(device)
                labels = item['labels'].to(device)
            
                outputs = model(images, boxes)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    test_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_predictions, 
        average='weighted'
    )
    
    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 主函数示例
def main():
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        root_dir='v1/',
        batch_size=1
    )
    
    # 创建模型
    model = UIElementDetector(num_classes=4)
    
    # 训练模型，添加resume参数
    train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=150,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        resume=True,  # 设置为True以启用断点续训
        checkpoint_path='best_model.pth'  # 指定检查点文件路径
    )

if __name__ == '__main__':
    main()