import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont


def _get_label_id(label):  
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

def visualize_bbox(image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 解析 XML 文件
    xml_path = image_path.replace('.png', '.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取 bndbox 和 name
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        label = _get_label_id(name)
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        if label == 0:  # clickable
            color = 'red'
            font_size = 20
        elif label == 1:  # selectable
            color = 'green'
            font_size = 20
        elif label == 2:  # scrollable
            color = 'blue'
            font_size = 20
        elif label == 3:  # disabled
            color = 'gray'
            font_size = 20
        elif label == 4:
            color = 'black'
            font_size = 20
        # 绘制边框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        # 打印名称
        draw.text((xmin, ymin), name, fill=color, font_size=font_size)

    # 保存或显示结果
    # image.show()  # 显示图片
    save_path = image_path.replace('.png', '_annotated.png')
    image.save(save_path)  # 保存图片

import os
dataset_dir = './dataset/train_dataset/jiguang'
for image_path in os.listdir(dataset_dir):
    if image_path.endswith('.png') and 'annotated' not in image_path:
        visualize_bbox(os.path.join(dataset_dir, image_path))
