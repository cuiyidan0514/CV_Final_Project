import random
from PIL import Image, ImageDraw, ImageFont

def draw_coordinates_boxes_on_image(image_path, coordinates, logits, labels, output_image_path, font_path, text_th=0.1, bbox_th=0.1, with_text=False):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    total_labels = len(labels)
    
    label_colors = {}
    font = ImageFont.truetype(font_path, int(height * 0.012))

    for i, (coord,label,logit) in enumerate(zip(coordinates,labels,logits)):
        # color = generate_color_from_hsv_pil(i, total_boxes)
        if label not in label_colors:
            new_color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)) 
            label_colors[label] = new_color
        else:
            new_color = label_colors[label]
        
        draw.rectangle(list(coord), outline=new_color, width=int(height * 0.0025))

        text_x = coord[0] + int(height * 0.0025)
        text_y = max(0, coord[1] - int(height * 0.013))
        text_to_draw = f"{label}:{logit:.2f}"
        draw.text((text_x, text_y), text_to_draw, fill=new_color, font=font)

    image = image.convert('RGB')
    image.save(f"{output_image_path}.png")
    # if with_text:
    #     image.save(f"{output_image_path}_with_text.png")
    # else:
    #     image.save(f"{output_image_path}.png")

def draw_boxes_in_format(image_path, coordinates, logits, labels, output_image_path, font_path, text_th=0.1, bbox_th=0.1, with_text=False):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
        
    label_colors = {}
    font = ImageFont.truetype(font_path, int(height * 0.012))
    set_click = labels['clickable']
    set_select = labels['selectable']
    set_scroll = labels['scrollable']
    
    for i, (coord,label,logit) in enumerate(zip(coordinates,labels,logits)):
        if label in set_click:
            label = 'clickable'
        elif label in set_select:
            label = 'selectable'
        elif label in set_scroll:
            label = 'scrollable'
        else:
            label = 'disabled'
        if label not in label_colors:
            new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
            label_colors[label] = new_color
        else:
            new_color = label_colors[label]

        draw.rectangle(coord, outline=new_color, width=int(height * 0.0025))

        text_x = coord[0] + int(height * 0.0025)
        text_y = max(0, coord[1] - int(height * 0.013))
        text_to_draw = f"{label}:{logit:.2f}"
        draw.text((text_x, text_y), text_to_draw, fill=new_color, font=font)

    image = image.convert('RGB')
    if with_text:
        image.save(f"{output_image_path}_with_text_in_format.png")
    else:
        image.save(f"{output_image_path}_in_format.png")



def text_draw_coordinates_boxes_on_image(image_path, coordinates, output_image_path):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for i, coord in enumerate(coordinates):
        draw.rectangle(coord, outline=new_color, width=int(height * 0.0025))

    image = image.convert('RGB')
    image.save(f"{output_image_path}_ocr.png")


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

def get_gt_label(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        label = _get_label_id(name)
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return label, xmin, ymin, xmax, ymax

def visualize_gt_bbox(image_path,save_path):
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
    image.save(save_path)  # 保存图片



def main():
    import os
    dataset_dir = './dataset/train_dataset/jiguang'
    for image_path in os.listdir(dataset_dir):
        if image_path.endswith('.png') and 'annotated' not in image_path:
            visualize_bbox(os.path.join(dataset_dir, image_path))

if __name__ == '__main__':
    main()

