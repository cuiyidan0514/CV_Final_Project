import cv2
import numpy as np
import os
import fnmatch


def connected_components_fill(image):  
    # 找到连通组件  
    num_labels, labels = cv2.connectedComponents(image)  
    # 创建一个空白图像，用于填充  
    filled_image = np.zeros_like(image)  
    # 遍历每个标记的区域  
    for label in range(1, num_labels):  
        filled_image[labels == label] = 255  # 将相应区域填充为白色  
    return filled_image 

def detect_root_area(src, tgt, pic):
    file = f"{src}{pic}.png"
    output_file = f"{tgt}{pic}/root.png"
    img2 = cv2.imread(file) # 待检测的图  
    if img2 is None:
        return
    ori_img2 = img2.copy()  

    # 检索前一张图
    parts = pic.split('_')
    file1 = None
    if len(parts) == 3: #frame_a_b
        prev = pic.split('_')[-1] #b
        for filename in os.listdir(src):
            if fnmatch.fnmatch(filename, f"frame_{prev}*"):
                file1 = f"{src}{filename}"
                break
    else: #frame_a
        file1 = f"{src}frame_1.png"
    
    if file1 is None:
        print(f"cannot find previous pic for {pic}")
        return
    
    img1 = cv2.imread(file1)

    # 确保两张图片大小相同  
    if img1.shape != img2.shape:  
        print("Error: The two images must have the same dimensions.")  
        return  

    diff = cv2.absdiff(img1, img2)   
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)   
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)  

    # 形态学操作以补全边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 定义内核  
    dilated_image = cv2.dilate(thresh, kernel, iterations=3)  # 膨胀  
    filled_image = connected_components_fill(dilated_image)  # 填充连通区域  

    contours, _ = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    if not contours:  
        print("No significant differences found.")  
        return  

    cv2.drawContours(img2, contours, -1, (0, 255, 0), 2)
    output_file2 = f"{tgt}{pic}/detect.png"
    cv2.imwrite(output_file2, img2)

    largest_contour = max(contours, key=cv2.contourArea)  
    x, y, w, h = cv2.boundingRect(largest_contour)  

    # 裁剪出不同区域  
    cropped_area = ori_img2[y:y+h, x:x+w]  

    # 保存裁剪后的图片  
    cv2.imwrite(output_file, cropped_area)  
    #print(f'root area saved as {output_file}.')  

    return x, y, x+w, y+h

if __name__ == "__main__":
    src = "./dataset/train_dataset/jiguang/"
    tgt = "./dataset/train_root_dataset/jiguang/"
    pic = "frame_40_29"
    detect_root_area(src, tgt, pic)