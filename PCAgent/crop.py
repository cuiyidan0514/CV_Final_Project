import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import clip
import torch


def crop_image(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst


def calculate_size(box):
    return (box[2]-box[0]) * (box[3]-box[1])


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea + 1e-9
    iou = interArea / unionArea
    
    return iou


def crop(image, box, i, text_data=None):
    image = Image.open(image)

    if text_data:
        draw = ImageDraw.Draw(image)
        draw.rectangle(((text_data[0], text_data[1]), (text_data[2], text_data[3])), outline="red", width=5)
        # font_size = int((text_data[3] - text_data[1])*0.75)
        # font = ImageFont.truetype("arial.ttf", font_size)
        # draw.text((text_data[0]+5, text_data[1]+5), str(i), font=font, fill="red")

    cropped_image = image.crop(box)
    cropped_image.save(f"./temp/{i}.jpg")
    

def in_box(box, target):
    if (box[0] > target[0]) and (box[1] > target[1]) and (box[2] < target[2]) and (box[3] < target[3]):
        return True
    else:
        return False

    
def crop_for_clip(image, box, i, position):
    image = Image.open(image)
    w, h = image.size
    if position == "left":
        bound = [0, 0, w/2, h]
    elif position == "right":
        bound = [w/2, 0, w, h]
    elif position == "top":
        bound = [0, 0, w, h/2]
    elif position == "bottom":
        bound = [0, h/2, w, h]
    elif position == "top left":
        bound = [0, 0, w/2, h/2]
    elif position == "top right":
        bound = [w/2, 0, w, h/2]
    elif position == "bottom left":
        bound = [0, h/2, w/2, h]
    elif position == "bottom right":
        bound = [w/2, h/2, w, h]
    else:
        bound = [0, 0, w, h]
    
    if in_box(box, bound):
        cropped_image = image.crop(box)
        cropped_image.save(f"./temp/{i}.jpg")
        return True
    else:
        return False
    
    
def clip_for_icon(clip_model, clip_preprocess, images, prompt):
    image_features = []
    for image_file in images:
        image = clip_preprocess(Image.open(image_file)).unsqueeze(0).to(next(clip_model.parameters()).device)
        image_feature = clip_model.encode_image(image)
        image_features.append(image_feature)
    image_features = torch.cat(image_features)
    
    text = clip.tokenize([prompt]).to(next(clip_model.parameters()).device)
    text_features = clip_model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=0).squeeze(0)
    _, max_pos = torch.max(similarity, dim=0)
    pos = max_pos.item()
    
    return pos

def split_image_dynamically(input_image_path, output_image_prefix, xmin, ymin, height_split=1, width_split=1):
    img = Image.open(input_image_path)
    width, height = img.size
    height_split = max(1, height_split)
    width_split = max(1, width_split)
    sub_width = width // width_split
    sub_height = height // height_split

    quadrants = []
    img_list = []
    img_x_list = []
    img_y_list = []

    for i in range(height_split):
        for j in range(width_split):
            left = j * sub_width
            upper = i * sub_height
            right = (j + 1) * sub_width if j < width_split - 1 else width
            lower = (i + 1) * sub_height if i < height_split - 1 else height

            box = (left, upper, right, lower)
            quadrants.append(box)

            sub_img = img.crop(box)
            sub_img_path = f"{output_image_prefix}_part_{i * width_split + j + 1}.png"
            sub_img.save(sub_img_path)
            img_list.append(sub_img_path)

            img_x_list.append(left + xmin)
            img_y_list.append(upper + ymin)

    return img_list, img_x_list, img_y_list

def split_image_into_4(input_image_path, output_image_prefix, xmin, ymin,mini=True):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 2
    sub_height = height // 2

    if mini:
        # crop into 4 sub images
        quadrants = [
            (0, 0, sub_width, sub_height),
            (sub_width, 0, width, sub_height),
            (0, sub_height, sub_width, height),
            (sub_width, sub_height, width, height)
        ]

        for i, box in enumerate(quadrants):
            sub_img = img.crop(box)
            sub_img.save(f"{output_image_prefix}_part_{i+1}.png")
        
        img_list = ['./screenshot/sub4/screenshot_part_1.png', './screenshot/sub4/screenshot_part_2.png',
                    './screenshot/sub4/screenshot_part_3.png', './screenshot/sub4/screenshot_part_4.png']
        img_x_list = [0, width/2, 0, width/2]
        img_y_list = [0, 0, height/2, height/2]
        img_x_list = [x + xmin for x in img_x_list]
        img_y_list = [y + ymin for y in img_y_list]
        return img_list, img_x_list, img_y_list
    else:
        quadrants = [
                (0, 0, sub_width, sub_height),
                (sub_width, 0, width, sub_height),
                (0, sub_height, sub_width, height),
                (sub_width, sub_height, width, height),
                (sub_width/2, 0, 3*sub_width/2, sub_height),
                (0, sub_height/2, sub_width, 3*sub_height/2),
                (sub_width/2, sub_height/2, 3*sub_width/2, 3*sub_height/2),
                (sub_width, sub_height/2, width, 3*sub_height/2),
                (sub_width/2, sub_height, 3*sub_width/2, height)
            ]

        for i, box in enumerate(quadrants):
            sub_img = img.crop(box)
            sub_img.save(f"{output_image_prefix}_part_{i+1}.png")
        
        img_list = ['./screenshot/sub4/screenshot_part_1.png', './screenshot/sub4/screenshot_part_2.png',
                    './screenshot/sub4/screenshot_part_3.png', './screenshot/sub4/screenshot_part_4.png',
                    './screenshot/sub4/screenshot_part_5.png', './screenshot/sub4/screenshot_part_6.png',
                    './screenshot/sub4/screenshot_part_7.png', './screenshot/sub4/screenshot_part_8.png',
                    './screenshot/sub4/screenshot_part_9.png']
        img_x_list = [0, width/2, 0, width/2, sub_width/2, 0, sub_width/2, sub_width, sub_width/2]
        img_y_list = [0, 0, height/2, height/2, 0, sub_height/2, sub_height/2, sub_height/2, sub_height]
        img_x_list = [x + xmin for x in img_x_list]
        img_y_list = [y + ymin for y in img_y_list]
        return img_list, img_x_list, img_y_list

# def split_image_into_4(input_image_path, output_image_prefix, xmin, ymin):
#     img = Image.open(input_image_path)
#     width, height = img.size

#     sub_width = width // 2
#     sub_height = height // 2

#     # crop into 4 sub images
#     quadrants = [
#         (0, 0, sub_width, sub_height),
#         (sub_width, 0, width, sub_height),
#         (0, sub_height, sub_width, height),
#         (sub_width, sub_height, width, height),
#         (sub_width/2, 0, 3*sub_width/2, sub_height),
#         (0, sub_height/2, sub_width, 3*sub_height/2),
#         (sub_width/2, sub_height/2, 3*sub_width/2, 3*sub_height/2),
#         (sub_width, sub_height/2, width, 3*sub_height/2),
#         (sub_width/2, sub_height, 3*sub_width/2, height)
#     ]

#     for i, box in enumerate(quadrants):
#         sub_img = img.crop(box)
#         sub_img.save(f"{output_image_prefix}_part_{i+1}.png")
    
#     img_list = ['./screenshot/sub4/screenshot_part_1.png', './screenshot/sub4/screenshot_part_2.png',
#                 './screenshot/sub4/screenshot_part_3.png', './screenshot/sub4/screenshot_part_4.png',
#                 './screenshot/sub4/screenshot_part_5.png', './screenshot/sub4/screenshot_part_6.png',
#                 './screenshot/sub4/screenshot_part_7.png', './screenshot/sub4/screenshot_part_8.png',
#                 './screenshot/sub4/screenshot_part_9.png']
#     img_x_list = [0, width/2, 0, width/2, sub_width/2, 0, sub_width/2, sub_width, sub_width/2]
#     img_y_list = [0, 0, height/2, height/2, 0, sub_height/2, sub_height/2, sub_height/2, sub_height]
#     img_x_list = [x + xmin for x in img_x_list]
#     img_y_list = [y + ymin for y in img_y_list]
#     return img_list, img_x_list, img_y_list

def split_image_into_9(input_image_path, output_image_prefix, xmin, ymin):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 3
    sub_height = height // 3

    # crop into 9 sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, width, sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, width, 2 * sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 2 * sub_height, width, height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")
    
    img_list = ['./screenshot/sub4/screenshot_part_1.png', './screenshot/sub4/screenshot_part_2.png',
                './screenshot/sub4/screenshot_part_3.png', './screenshot/sub4/screenshot_part_4.png',
                './screenshot/sub4/screenshot_part_5.png', './screenshot/sub4/screenshot_part_6.png',
                './screenshot/sub4/screenshot_part_7.png', './screenshot/sub4/screenshot_part_8.png',
                './screenshot/sub4/screenshot_part_9.png']
    img_x_list = [0, width/2, 0, width/2, sub_width/2, 0, sub_width/2, sub_width, sub_width/2]
    img_y_list = [0, 0, height/2, height/2, 0, sub_height/2, sub_height/2, sub_height/2, sub_height]
    
    return img_list, img_x_list, img_y_list

def split_image_into_16(input_image_path, output_image_prefix):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 4
    sub_height = height // 4

    # crop into 16 sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 1 * width,     sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 1 * width,     2 * sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 1 * width,     3 * sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 3 * sub_height, 1 * width,     height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub16/screenshot16_part_1.png','./screenshot/sub16/screenshot16_part_2.png','./screenshot/sub16/screenshot16_part_3.png','./screenshot/sub16/screenshot16_part_4.png',
                './screenshot/sub16/screenshot16_part_5.png','./screenshot/sub16/screenshot16_part_6.png','./screenshot/sub16/screenshot16_part_7.png','./screenshot/sub16/screenshot16_part_8.png',
                './screenshot/sub16/screenshot16_part_9.png','./screenshot/sub16/screenshot16_part_10.png','./screenshot/sub16/screenshot16_part_11.png','./screenshot/sub16/screenshot16_part_12.png',
                './screenshot/sub16/screenshot16_part_13.png','./screenshot/sub16/screenshot16_part_14.png','./screenshot/sub16/screenshot16_part_15.png','./screenshot/sub16/screenshot16_part_16.png',]
    total_width, total_height = width, height
    img_x_list = [0, total_width/4, total_width/2, 3*total_width/4,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  0, total_width/4, total_width/2, 3*total_width/4]
    img_y_list = [0, 0, 0, 0,
                  total_height/4, total_height/4, total_height/4, total_height/4,
                  total_height/2, total_height/2, total_height/2, total_height/2,
                  3*total_height/4, 3*total_height/4, 3*total_height/4, 3*total_height/4]
    return img_list, img_x_list, img_y_list

def split_image_into_25(input_image_path, output_image_prefix):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 5
    sub_height = height // 5

    # crop into 25 sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 4 * sub_width, sub_height),
        (4 * sub_width, 0, 1 * width, sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 4 * sub_width, 2 * sub_height),
        (4 * sub_width, sub_height, 1 * width, 2 * sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 4 * sub_width, 3 * sub_height),
        (4 * sub_width, 2 * sub_height, 1 * width,     3 * sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, 4 * sub_height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, 4 * sub_height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, 4 * sub_height),
        (3 * sub_width, 3 * sub_height, 4 * sub_width, 4 * sub_height),
        (4 * sub_width, 3 * sub_height, 1 * width,     4 * sub_height),

        (0 * sub_width, 4 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 4 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 4 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 4 * sub_height, 4 * sub_width, height),
        (4 * sub_width, 4 * sub_height, 1 * width,     height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub25/screenshot25_part_1.png','./screenshot/sub25/screenshot25_part_2.png','./screenshot/sub25/screenshot25_part_3.png','./screenshot/sub25/screenshot25_part_4.png',
                './screenshot/sub25/screenshot25_part_5.png','./screenshot/sub25/screenshot25_part_6.png','./screenshot/sub25/screenshot25_part_7.png','./screenshot/sub25/screenshot25_part_8.png',
                './screenshot/sub25/screenshot25_part_9.png','./screenshot/sub25/screenshot25_part_10.png','./screenshot/sub25/screenshot25_part_11.png','./screenshot/sub25/screenshot25_part_12.png',
                './screenshot/sub25/screenshot25_part_13.png','./screenshot/sub25/screenshot25_part_14.png','./screenshot/sub25/screenshot25_part_15.png','./screenshot/sub25/screenshot25_part_16.png',
                './screenshot/sub25/screenshot25_part_17.png','./screenshot/sub25/screenshot25_part_18.png','./screenshot/sub25/screenshot25_part_19.png','./screenshot/sub25/screenshot25_part_20.png',
                './screenshot/sub25/screenshot25_part_21.png','./screenshot/sub25/screenshot25_part_22.png','./screenshot/sub25/screenshot25_part_23.png','./screenshot/sub25/screenshot25_part_24.png',
                './screenshot/sub25/screenshot25_part_25.png']
    total_width, total_height = width, height
    img_x_list = [0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5]
    img_y_list = [0, 0, 0, 0, 0,
                  total_height/5, total_height/5, total_height/5, total_height/5, total_height/5,
                  2*total_height/5, 2*total_height/5, 2*total_height/5, 2*total_height/5, 2*total_height/5,
                  3*total_height/5, 3*total_height/5, 3*total_height/5, 3*total_height/5, 3*total_height/5, 
                  4*total_height/5, 4*total_height/5, 4*total_height/5, 4*total_height/5, 4*total_height/5]
    return img_list, img_x_list, img_y_list

def split_image_into_36(input_image_path, output_image_prefix):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 6
    sub_height = height // 6

    # crop into 36 sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 4 * sub_width, sub_height),
        (4 * sub_width, 0, 5 * sub_width, sub_height),
        (5 * sub_width, 0, width, sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 4 * sub_width, 2 * sub_height),
        (4 * sub_width, sub_height, 5 * sub_width, 2 * sub_height),
        (5 * sub_width, sub_height, width, 2 * sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 4 * sub_width, 3 * sub_height),
        (4 * sub_width, 2 * sub_height, 5 * sub_width, 3 * sub_height),
        (5 * sub_width, 2 * sub_height, width, 3 * sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, 4 * sub_height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, 4 * sub_height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, 4 * sub_height),
        (3 * sub_width, 3 * sub_height, 4 * sub_width, 4 * sub_height),
        (4 * sub_width, 3 * sub_height, 5 * sub_width, 4 * sub_height),
        (5 * sub_width, 3 * sub_height, width, 4 * sub_height),

        (0 * sub_width, 4 * sub_height, 1 * sub_width, 5 * sub_height),
        (1 * sub_width, 4 * sub_height, 2 * sub_width, 5 * sub_height),
        (2 * sub_width, 4 * sub_height, 3 * sub_width, 5 * sub_height),
        (3 * sub_width, 4 * sub_height, 4 * sub_width, 5 * sub_height),
        (4 * sub_width, 4 * sub_height, 5 * sub_width, 5 * sub_height),
        (5 * sub_width, 4 * sub_height, width, 5 * sub_height),

        (0 * sub_width, 5 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 5 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 5 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 5 * sub_height, 4 * sub_width, height),
        (4 * sub_width, 5 * sub_height, 5 * sub_width, height),
        (5 * sub_width, 5 * sub_height, width, height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub36/screenshot36_part_1.png','./screenshot/sub36/screenshot36_part_2.png','./screenshot/sub36/screenshot36_part_3.png','./screenshot/sub36/screenshot36_part_4.png',
                './screenshot/sub36/screenshot36_part_5.png','./screenshot/sub36/screenshot36_part_6.png','./screenshot/sub36/screenshot36_part_7.png','./screenshot/sub36/screenshot36_part_8.png',
                './screenshot/sub36/screenshot36_part_9.png','./screenshot/sub36/screenshot36_part_10.png','./screenshot/sub36/screenshot36_part_11.png','./screenshot/sub36/screenshot36_part_12.png',
                './screenshot/sub36/screenshot36_part_13.png','./screenshot/sub36/screenshot36_part_14.png','./screenshot/sub36/screenshot36_part_15.png','./screenshot/sub36/screenshot36_part_16.png',
                './screenshot/sub36/screenshot36_part_17.png','./screenshot/sub36/screenshot36_part_18.png','./screenshot/sub36/screenshot36_part_19.png','./screenshot/sub36/screenshot36_part_20.png',
                './screenshot/sub36/screenshot36_part_21.png','./screenshot/sub36/screenshot36_part_22.png','./screenshot/sub36/screenshot36_part_23.png','./screenshot/sub36/screenshot36_part_24.png',
                './screenshot/sub36/screenshot36_part_25.png','./screenshot/sub36/screenshot36_part_26.png','./screenshot/sub36/screenshot36_part_27.png','./screenshot/sub36/screenshot36_part_28.png',
                './screenshot/sub36/screenshot36_part_29.png','./screenshot/sub36/screenshot36_part_30.png','./screenshot/sub36/screenshot36_part_31.png','./screenshot/sub36/screenshot36_part_32.png',
                './screenshot/sub36/screenshot36_part_33.png','./screenshot/sub36/screenshot36_part_34.png','./screenshot/sub36/screenshot36_part_35.png','./screenshot/sub36/screenshot36_part_36.png']
    total_width = width
    total_height = height
    img_x_list = [0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6]

    img_y_list = [0, 0, 0, 0, 0, 0,
                  total_height/6, total_height/6, total_height/6, total_height/6, total_height/6, total_height/6,
                  2*total_height/6, 2*total_height/6, 2*total_height/6, 2*total_height/6, 2*total_height/6, 2*total_height/6,
                  3*total_height/6, 3*total_height/6, 3*total_height/6, 3*total_height/6, 3*total_height/6, 3*total_height/6,
                  4*total_height/6, 4*total_height/6, 4*total_height/6, 4*total_height/6, 4*total_height/6, 4*total_height/6,
                  5*total_height/6, 5*total_height/6, 5*total_height/6, 5*total_height/6, 5*total_height/6, 5*total_height/6]
    return img_list, img_x_list, img_y_list

def split_image_into_16_and_shift(input_image_path, output_image_prefix):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 4
    sub_height = height // 4
    shift_w = sub_width // 2
    shift_h = sub_height // 2

    # crop into 16 sub images and 9 shift sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 1 * width,     sub_height),

        (shift_w, shift_h, shift_w+sub_width, shift_h+sub_height),
        (shift_w+sub_width, shift_h, shift_w+2*sub_width, shift_h+sub_height),
        (shift_w+2*sub_width, shift_h, shift_w+3*sub_width, shift_h+sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 1 * width,     2 * sub_height),

        (shift_w, shift_h+sub_height, shift_w+sub_width, shift_h+2*sub_height),
        (shift_w+sub_width, shift_h+sub_height, shift_w+2*sub_width, shift_h+2*sub_height),
        (shift_w+2*sub_width, shift_h+sub_height, shift_w+3*sub_width, shift_h+2*sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 1 * width,     3 * sub_height),

        (shift_w, shift_h+2*sub_height, shift_w+sub_width, shift_h+3*sub_height),
        (shift_w+sub_width, shift_h+2*sub_height, shift_w+2*sub_width, shift_h+3*sub_height),
        (shift_w+2*sub_width, shift_h+2*sub_height, shift_w+3*sub_width, shift_h+3*sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 3 * sub_height, 1 * width,     height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub16shift/screenshot16_part_1.png','./screenshot/sub16shift/screenshot16_part_2.png','./screenshot/sub16shift/screenshot16_part_3.png','./screenshot/sub16shift/screenshot16_part_4.png',
                './screenshot/sub16shift/screenshot16_part_5.png','./screenshot/sub16shift/screenshot16_part_6.png','./screenshot/sub16shift/screenshot16_part_7.png','./screenshot/sub16shift/screenshot16_part_8.png',
                './screenshot/sub16shift/screenshot16_part_9.png','./screenshot/sub16shift/screenshot16_part_10.png','./screenshot/sub16shift/screenshot16_part_11.png','./screenshot/sub16shift/screenshot16_part_12.png',
                './screenshot/sub16shift/screenshot16_part_13.png','./screenshot/sub16shift/screenshot16_part_14.png','./screenshot/sub16shift/screenshot16_part_15.png','./screenshot/sub16shift/screenshot16_part_16.png',
                './screenshot/sub16shift/screenshot16_part_17.png','./screenshot/sub16shift/screenshot16_part_18.png','./screenshot/sub16shift/screenshot16_part_19.png','./screenshot/sub16shift/screenshot16_part_20.png',
                './screenshot/sub16shift/screenshot16_part_21.png','./screenshot/sub16shift/screenshot16_part_22.png','./screenshot/sub16shift/screenshot16_part_23.png','./screenshot/sub16shift/screenshot16_part_24.png',
                './screenshot/sub16shift/screenshot16_part_25.png']
    total_width, total_height = width, height
    img_x_list = [0, total_width/4, total_width/2, 3*total_width/4,
                  total_width/8, 3*total_width/8, 5*total_width/8,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  total_width/8, 3*total_width/8, 5*total_width/8,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  total_width/8, 3*total_width/8, 5*total_width/8,
                  0, total_width/4, total_width/2, 3*total_width/4]
    img_y_list = [0, 0, 0, 0,
                  total_height/8, total_height/8, total_height/8,
                  total_height/4, total_height/4, total_height/4, total_height/4,
                  3*total_height/8, 3*total_height/8, 3*total_height/8,
                  total_height/2, total_height/2, total_height/2, total_height/2,
                  5*total_height/8, 5*total_height/8, 5*total_height/8,
                  3*total_height/4, 3*total_height/4, 3*total_height/4, 3*total_height/4]
    return img_list, img_x_list, img_y_list






    img_list = ['./screenshot/sub9/screenshot9_part_1.png','./screenshot/sub9/screenshot9_part_2.png','./screenshot/sub9/screenshot9_part_3.png','./screenshot/sub9/screenshot9_part_4.png',
                './screenshot/sub9/screenshot9_part_5.png','./screenshot/sub9/screenshot9_part_6.png','./screenshot/sub9/screenshot9_part_7.png','./screenshot/sub9/screenshot9_part_8.png',
                './screenshot/sub9/screenshot9_part_9.png']
    total_width, total_height = width, height
    img_x_list = [0, total_width/3, 2*total_width/3,
                  0, total_width/3, 2*total_width/3,
                  0, total_width/3, 2*total_width/3]
    img_y_list = [0, 0, 0,
                  total_height/3, total_height/3, total_height/3, 
                  2*total_height/3, 2*total_height/3, 2*total_height/3]
    img_x_list = [x + xmin for x in img_x_list]
    img_y_list = [y + ymin for y in img_y_list]
    return img_list, img_x_list, img_y_list

def split_image_into_16(input_image_path, output_image_prefix, xmin, ymin):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 4
    sub_height = height // 4

    # crop into 16 sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 1 * width,     sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 1 * width,     2 * sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 1 * width,     3 * sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 3 * sub_height, 1 * width,     height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub16/screenshot16_part_1.png','./screenshot/sub16/screenshot16_part_2.png','./screenshot/sub16/screenshot16_part_3.png','./screenshot/sub16/screenshot16_part_4.png',
                './screenshot/sub16/screenshot16_part_5.png','./screenshot/sub16/screenshot16_part_6.png','./screenshot/sub16/screenshot16_part_7.png','./screenshot/sub16/screenshot16_part_8.png',
                './screenshot/sub16/screenshot16_part_9.png','./screenshot/sub16/screenshot16_part_10.png','./screenshot/sub16/screenshot16_part_11.png','./screenshot/sub16/screenshot16_part_12.png',
                './screenshot/sub16/screenshot16_part_13.png','./screenshot/sub16/screenshot16_part_14.png','./screenshot/sub16/screenshot16_part_15.png','./screenshot/sub16/screenshot16_part_16.png',]
    total_width, total_height = width, height
    img_x_list = [0, total_width/4, total_width/2, 3*total_width/4,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  0, total_width/4, total_width/2, 3*total_width/4]
    img_y_list = [0, 0, 0, 0,
                  total_height/4, total_height/4, total_height/4, total_height/4,
                  total_height/2, total_height/2, total_height/2, total_height/2,
                  3*total_height/4, 3*total_height/4, 3*total_height/4, 3*total_height/4]
    img_x_list = [x + xmin for x in img_x_list]
    img_y_list = [y + ymin for y in img_y_list]
    return img_list, img_x_list, img_y_list

def split_image_into_25(input_image_path, output_image_prefix, xmin, ymin):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 5
    sub_height = height // 5

    # crop into 25 sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 4 * sub_width, sub_height),
        (4 * sub_width, 0, 1 * width, sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 4 * sub_width, 2 * sub_height),
        (4 * sub_width, sub_height, 1 * width, 2 * sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 4 * sub_width, 3 * sub_height),
        (4 * sub_width, 2 * sub_height, 1 * width,     3 * sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, 4 * sub_height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, 4 * sub_height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, 4 * sub_height),
        (3 * sub_width, 3 * sub_height, 4 * sub_width, 4 * sub_height),
        (4 * sub_width, 3 * sub_height, 1 * width,     4 * sub_height),

        (0 * sub_width, 4 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 4 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 4 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 4 * sub_height, 4 * sub_width, height),
        (4 * sub_width, 4 * sub_height, 1 * width,     height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub25/screenshot25_part_1.png','./screenshot/sub25/screenshot25_part_2.png','./screenshot/sub25/screenshot25_part_3.png','./screenshot/sub25/screenshot25_part_4.png',
                './screenshot/sub25/screenshot25_part_5.png','./screenshot/sub25/screenshot25_part_6.png','./screenshot/sub25/screenshot25_part_7.png','./screenshot/sub25/screenshot25_part_8.png',
                './screenshot/sub25/screenshot25_part_9.png','./screenshot/sub25/screenshot25_part_10.png','./screenshot/sub25/screenshot25_part_11.png','./screenshot/sub25/screenshot25_part_12.png',
                './screenshot/sub25/screenshot25_part_13.png','./screenshot/sub25/screenshot25_part_14.png','./screenshot/sub25/screenshot25_part_15.png','./screenshot/sub25/screenshot25_part_16.png',
                './screenshot/sub25/screenshot25_part_17.png','./screenshot/sub25/screenshot25_part_18.png','./screenshot/sub25/screenshot25_part_19.png','./screenshot/sub25/screenshot25_part_20.png',
                './screenshot/sub25/screenshot25_part_21.png','./screenshot/sub25/screenshot25_part_22.png','./screenshot/sub25/screenshot25_part_23.png','./screenshot/sub25/screenshot25_part_24.png',
                './screenshot/sub25/screenshot25_part_25.png']
    total_width, total_height = width, height
    img_x_list = [0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5,
                  0, total_width/5, 2*total_width/5, 3*total_width/5, 4*total_width/5]
    img_y_list = [0, 0, 0, 0, 0,
                  total_height/5, total_height/5, total_height/5, total_height/5, total_height/5,
                  2*total_height/5, 2*total_height/5, 2*total_height/5, 2*total_height/5, 2*total_height/5,
                  3*total_height/5, 3*total_height/5, 3*total_height/5, 3*total_height/5, 3*total_height/5, 
                  4*total_height/5, 4*total_height/5, 4*total_height/5, 4*total_height/5, 4*total_height/5]
    img_x_list = [x + xmin for x in img_x_list]
    img_y_list = [y + ymin for y in img_y_list]
    return img_list, img_x_list, img_y_list

def split_image_into_36(input_image_path, output_image_prefix, xmin, ymin):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 6
    sub_height = height // 6

    # crop into 36 sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 4 * sub_width, sub_height),
        (4 * sub_width, 0, 5 * sub_width, sub_height),
        (5 * sub_width, 0, width, sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 4 * sub_width, 2 * sub_height),
        (4 * sub_width, sub_height, 5 * sub_width, 2 * sub_height),
        (5 * sub_width, sub_height, width, 2 * sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 4 * sub_width, 3 * sub_height),
        (4 * sub_width, 2 * sub_height, 5 * sub_width, 3 * sub_height),
        (5 * sub_width, 2 * sub_height, width, 3 * sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, 4 * sub_height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, 4 * sub_height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, 4 * sub_height),
        (3 * sub_width, 3 * sub_height, 4 * sub_width, 4 * sub_height),
        (4 * sub_width, 3 * sub_height, 5 * sub_width, 4 * sub_height),
        (5 * sub_width, 3 * sub_height, width, 4 * sub_height),

        (0 * sub_width, 4 * sub_height, 1 * sub_width, 5 * sub_height),
        (1 * sub_width, 4 * sub_height, 2 * sub_width, 5 * sub_height),
        (2 * sub_width, 4 * sub_height, 3 * sub_width, 5 * sub_height),
        (3 * sub_width, 4 * sub_height, 4 * sub_width, 5 * sub_height),
        (4 * sub_width, 4 * sub_height, 5 * sub_width, 5 * sub_height),
        (5 * sub_width, 4 * sub_height, width, 5 * sub_height),

        (0 * sub_width, 5 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 5 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 5 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 5 * sub_height, 4 * sub_width, height),
        (4 * sub_width, 5 * sub_height, 5 * sub_width, height),
        (5 * sub_width, 5 * sub_height, width, height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub36/screenshot36_part_1.png','./screenshot/sub36/screenshot36_part_2.png','./screenshot/sub36/screenshot36_part_3.png','./screenshot/sub36/screenshot36_part_4.png',
                './screenshot/sub36/screenshot36_part_5.png','./screenshot/sub36/screenshot36_part_6.png','./screenshot/sub36/screenshot36_part_7.png','./screenshot/sub36/screenshot36_part_8.png',
                './screenshot/sub36/screenshot36_part_9.png','./screenshot/sub36/screenshot36_part_10.png','./screenshot/sub36/screenshot36_part_11.png','./screenshot/sub36/screenshot36_part_12.png',
                './screenshot/sub36/screenshot36_part_13.png','./screenshot/sub36/screenshot36_part_14.png','./screenshot/sub36/screenshot36_part_15.png','./screenshot/sub36/screenshot36_part_16.png',
                './screenshot/sub36/screenshot36_part_17.png','./screenshot/sub36/screenshot36_part_18.png','./screenshot/sub36/screenshot36_part_19.png','./screenshot/sub36/screenshot36_part_20.png',
                './screenshot/sub36/screenshot36_part_21.png','./screenshot/sub36/screenshot36_part_22.png','./screenshot/sub36/screenshot36_part_23.png','./screenshot/sub36/screenshot36_part_24.png',
                './screenshot/sub36/screenshot36_part_25.png','./screenshot/sub36/screenshot36_part_26.png','./screenshot/sub36/screenshot36_part_27.png','./screenshot/sub36/screenshot36_part_28.png',
                './screenshot/sub36/screenshot36_part_29.png','./screenshot/sub36/screenshot36_part_30.png','./screenshot/sub36/screenshot36_part_31.png','./screenshot/sub36/screenshot36_part_32.png',
                './screenshot/sub36/screenshot36_part_33.png','./screenshot/sub36/screenshot36_part_34.png','./screenshot/sub36/screenshot36_part_35.png','./screenshot/sub36/screenshot36_part_36.png']
    total_width = width
    total_height = height
    img_x_list = [0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6,
                  0, total_width/6, 2*total_width/6, 3*total_width/6, 4*total_width/6, 5*total_width/6]

    img_y_list = [0, 0, 0, 0, 0, 0,
                  total_height/6, total_height/6, total_height/6, total_height/6, total_height/6, total_height/6,
                  2*total_height/6, 2*total_height/6, 2*total_height/6, 2*total_height/6, 2*total_height/6, 2*total_height/6,
                  3*total_height/6, 3*total_height/6, 3*total_height/6, 3*total_height/6, 3*total_height/6, 3*total_height/6,
                  4*total_height/6, 4*total_height/6, 4*total_height/6, 4*total_height/6, 4*total_height/6, 4*total_height/6,
                  5*total_height/6, 5*total_height/6, 5*total_height/6, 5*total_height/6, 5*total_height/6, 5*total_height/6]
    img_x_list = [x + xmin for x in img_x_list]
    img_y_list = [y + ymin for y in img_y_list]
    return img_list, img_x_list, img_y_list

def split_image_into_16_and_shift(input_image_path, output_image_prefix, xmin, ymin):
    img = Image.open(input_image_path)
    width, height = img.size

    sub_width = width // 4
    sub_height = height // 4
    shift_w = sub_width // 2
    shift_h = sub_height // 2

    # crop into 16 sub images and 9 shift sub images
    quadrants = [
        (0 * sub_width, 0, 1 * sub_width, sub_height),
        (1 * sub_width, 0, 2 * sub_width, sub_height),
        (2 * sub_width, 0, 3 * sub_width, sub_height),
        (3 * sub_width, 0, 1 * width,     sub_height),

        (shift_w, shift_h, shift_w+sub_width, shift_h+sub_height),
        (shift_w+sub_width, shift_h, shift_w+2*sub_width, shift_h+sub_height),
        (shift_w+2*sub_width, shift_h, shift_w+3*sub_width, shift_h+sub_height),
        
        (0 * sub_width, sub_height, 1 * sub_width, 2 * sub_height),
        (1 * sub_width, sub_height, 2 * sub_width, 2 * sub_height),
        (2 * sub_width, sub_height, 3 * sub_width, 2 * sub_height),
        (3 * sub_width, sub_height, 1 * width,     2 * sub_height),

        (shift_w, shift_h+sub_height, shift_w+sub_width, shift_h+2*sub_height),
        (shift_w+sub_width, shift_h+sub_height, shift_w+2*sub_width, shift_h+2*sub_height),
        (shift_w+2*sub_width, shift_h+sub_height, shift_w+3*sub_width, shift_h+2*sub_height),

        (0 * sub_width, 2 * sub_height, 1 * sub_width, 3 * sub_height),
        (1 * sub_width, 2 * sub_height, 2 * sub_width, 3 * sub_height),
        (2 * sub_width, 2 * sub_height, 3 * sub_width, 3 * sub_height),
        (3 * sub_width, 2 * sub_height, 1 * width,     3 * sub_height),

        (shift_w, shift_h+2*sub_height, shift_w+sub_width, shift_h+3*sub_height),
        (shift_w+sub_width, shift_h+2*sub_height, shift_w+2*sub_width, shift_h+3*sub_height),
        (shift_w+2*sub_width, shift_h+2*sub_height, shift_w+3*sub_width, shift_h+3*sub_height),

        (0 * sub_width, 3 * sub_height, 1 * sub_width, height),
        (1 * sub_width, 3 * sub_height, 2 * sub_width, height),
        (2 * sub_width, 3 * sub_height, 3 * sub_width, height),
        (3 * sub_width, 3 * sub_height, 1 * width,     height),
    ]

    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

    img_list = ['./screenshot/sub16shift/screenshot16_part_1.png','./screenshot/sub16shift/screenshot16_part_2.png','./screenshot/sub16shift/screenshot16_part_3.png','./screenshot/sub16shift/screenshot16_part_4.png',
                './screenshot/sub16shift/screenshot16_part_5.png','./screenshot/sub16shift/screenshot16_part_6.png','./screenshot/sub16shift/screenshot16_part_7.png','./screenshot/sub16shift/screenshot16_part_8.png',
                './screenshot/sub16shift/screenshot16_part_9.png','./screenshot/sub16shift/screenshot16_part_10.png','./screenshot/sub16shift/screenshot16_part_11.png','./screenshot/sub16shift/screenshot16_part_12.png',
                './screenshot/sub16shift/screenshot16_part_13.png','./screenshot/sub16shift/screenshot16_part_14.png','./screenshot/sub16shift/screenshot16_part_15.png','./screenshot/sub16shift/screenshot16_part_16.png',
                './screenshot/sub16shift/screenshot16_part_17.png','./screenshot/sub16shift/screenshot16_part_18.png','./screenshot/sub16shift/screenshot16_part_19.png','./screenshot/sub16shift/screenshot16_part_20.png',
                './screenshot/sub16shift/screenshot16_part_21.png','./screenshot/sub16shift/screenshot16_part_22.png','./screenshot/sub16shift/screenshot16_part_23.png','./screenshot/sub16shift/screenshot16_part_24.png',
                './screenshot/sub16shift/screenshot16_part_25.png']
    total_width, total_height = width, height
    img_x_list = [0, total_width/4, total_width/2, 3*total_width/4,
                  total_width/8, 3*total_width/8, 5*total_width/8,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  total_width/8, 3*total_width/8, 5*total_width/8,
                  0, total_width/4, total_width/2, 3*total_width/4,
                  total_width/8, 3*total_width/8, 5*total_width/8,
                  0, total_width/4, total_width/2, 3*total_width/4]
    img_y_list = [0, 0, 0, 0,
                  total_height/8, total_height/8, total_height/8,
                  total_height/4, total_height/4, total_height/4, total_height/4,
                  3*total_height/8, 3*total_height/8, 3*total_height/8,
                  total_height/2, total_height/2, total_height/2, total_height/2,
                  5*total_height/8, 5*total_height/8, 5*total_height/8,
                  3*total_height/4, 3*total_height/4, 3*total_height/4, 3*total_height/4]
    img_x_list = [x + xmin for x in img_x_list]
    img_y_list = [y + ymin for y in img_y_list]
    return img_list, img_x_list, img_y_list
