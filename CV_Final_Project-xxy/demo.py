import os  
import xml.etree.ElementTree as ET  
import xml.dom.minidom as minidom  
import cv2  
import numpy as np  
from typing import List, Dict, Tuple  
import matplotlib.pyplot as plt 

class AndroidElement:  
    def __init__(self, uid: str, bbox: Tuple[Tuple[int, int], Tuple[int, int]], attrib: str):  
        self.uid = uid  
        self.bbox = bbox # bounding box的对角点坐标 
        self.attrib = attrib  

from models import GroundingDINO
class BBoxGenerator:
    def __init__(self, model: GroundingDINO):
        self.model = model

    def generate_bboxes(self, image: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        results = self.model(image,configs = 'all object')
        return results


def parse_screenshot_layout(screenshot_path: str, output_dir: str, configs: Dict = None) -> Dict:  
    """  
    解析屏幕截图并生成标准布局文件  
    
    Args:  
        screenshot_path (str): 屏幕截图路径  
        output_dir (str): 输出目录  
        configs (Dict, optional): 配置参数  
    
    Returns:  
        Dict: 包含布局信息的字典  
    """  
    # 默认配置  
    default_configs = {  
        "MIN_DIST": 50,  # 元素去重最小距离  
        "DARK_MODE": False  # 是否为深色模式  
    }  
    configs = configs or default_configs

    # 获取原始文件名和扩展名  
    original_filename = os.path.basename(screenshot_path)  
    filename, ext = os.path.splitext(original_filename)  

    # 生成新的文件名  
    labeled_image_filename = f"labeled_{filename}{ext}"  
    labeled_xml_filename = f"labeled_{filename}.xml" 

    # 完整路径  
    labeled_path = os.path.join(output_dir, labeled_image_filename)  
    layout_path = os.path.join(output_dir, labeled_xml_filename)  

    # 使用OpenCV加载图像  
    image = cv2.imread(screenshot_path)  
    height, width = image.shape[:2]  

    # 对截图中的UI组件元素进行解析，并生成列表  
    elem_list = _collect_ui_elements(  
        image,     
        configs  
    )  

    # 根据解析得到的elem_list生成标注图   
    _draw_bbox_multi(  
        image,   
        labeled_path,   
        elem_list,   
        dark_mode=configs["DARK_MODE"],
        show_image=True  
    )  

    # 生成XML布局文件  
    layout_xml = _generate_layout_xml(screenshot_path, elem_list, width, height)  
    
    with open(layout_path, 'w', encoding='utf-8') as f:  
        f.write(layout_xml)  

    return {  
        "screenshot": screenshot_path,  
        "labeled_screenshot": labeled_path,  
        "layout_xml": layout_path,  
        "elements": elem_list  
    }  

def _collect_ui_elements(  
    image: np.ndarray,     
    configs: Dict  
) -> List[AndroidElement]:  
    """  
    收集UI元素并去重  
    """  
    # 这里需要添加更复杂的元素检测逻辑：PaddleOCR/YOLO/OSworld/groundingDINO  
    # 目前使用简单的边缘检测作为测试  
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    # edges = cv2.Canny(gray, 50, 150)  
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    contours = bbox_generator.generate_bboxes(image)
    elem_list = []  
    for i, cnt in enumerate(contours): #获取一系列标定框 
        x, y, w, h = cv2.boundingRect(cnt)  
        
        # 过滤掉太小的轮廓  
        if w * h < 100:  
            continue  
        
        bbox = ((x, y), (x+w, y+h))  
        center = (x + w // 2, y + h // 2)  
        
        # 去重策略，如果两个框距离太近就舍弃一个  
        close = any(  
            ((e.bbox[0][0] + e.bbox[1][0]) // 2 - center[0]) ** 2 +   
            ((e.bbox[0][1] + e.bbox[1][1]) // 2 - center[1]) ** 2   
            <= configs["MIN_DIST"] ** 2  
            for e in elem_list  
        )  
        if not close:  
            elem = AndroidElement(  
                uid=f"elem_{i}",   
                bbox=bbox,   
                attrib="clickable"  # 先简化处理，全都假定是clickable  
            )  
            elem_list.append(elem)  
    
    return elem_list  

def _draw_bbox_multi(  
    image: np.ndarray,   
    output_path: str,   
    elem_list: List[AndroidElement],   
    dark_mode: bool = False,
    show_image: bool = True  
):  
    """  
    在图像上绘制边界框  
    """  
    for i, elem in enumerate(elem_list):  
        x1, y1 = elem.bbox[0]  
        x2, y2 = elem.bbox[1]  
        
        # 根据深色/浅色模式选择颜色  
        color = (0, 255, 0) if not dark_mode else (0, 255, 255)  
        # 绘制矩形  
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  
        # 添加编号  
        cv2.putText(  
            image,   
            str(i+1),   
            (x1, y1-10),   
            cv2.FONT_HERSHEY_SIMPLEX,   
            0.9,   
            color,   
            2  
        )  
    cv2.imwrite(output_path, image)  

    #显示图像
    if show_image:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
        plt.figure(figsize=(15, 10))  
        plt.imshow(image_rgb)  
        plt.axis('on')  # 显示坐标轴  
        plt.title('Annotated UI Elements')  
        plt.tight_layout()  
        plt.show()

def _generate_layout_xml(
    screenshot_path: str,  
    elem_list: List[AndroidElement],   
    width: int,   
    height: int  
) -> str:  
    """  
    根据训练集的xml生成标准XML布局文件  
    """
    # 创建根节点
    annotation = ET.Element('annotation')  
    
    # 文件信息
    folder = ET.SubElement(annotation, 'folder')  
    folder.text = os.path.basename(os.path.dirname(screenshot_path))  
    
    filename = ET.SubElement(annotation, 'filename')  
    filename.text = os.path.basename(screenshot_path)  
    
    path = ET.SubElement(annotation, 'path')  
    path.text = os.path.abspath(screenshot_path)  

    # 源信息
    source = ET.SubElement(annotation, 'source')  
    database = ET.SubElement(source, 'database')  
    database.text = 'Unknown'

    # 图像尺寸信息
    size = ET.SubElement(annotation, 'size')  
    width_elem = ET.SubElement(size, 'width')  
    width_elem.text = str(width)  
    height_elem = ET.SubElement(size, 'height')  
    height_elem.text = str(height)  
    depth = ET.SubElement(size, 'depth')  
    depth.text = '3'

    segmented = ET.SubElement(annotation, 'segmented')  
    segmented.text = '0' 
    
    for i, elem in enumerate(elem_list, 1): 
        obj = ET.SubElement(annotation, 'object')  
 
        name = ET.SubElement(obj, 'name')  
        name.text = elem.attrib

        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(obj, 'truncated')  
        truncated.text = '0'  
        
        difficult = ET.SubElement(obj, 'difficult')  
        difficult.text = '0'
        
        # 位置信息  
        bndbox = ET.SubElement(obj, 'bndbox')  
        xmin = ET.SubElement(bndbox, 'xmin')  
        xmin.text = str(elem.bbox[0][0])  
        ymin = ET.SubElement(bndbox, 'ymin')  
        ymin.text = str(elem.bbox[0][1])  
        xmax = ET.SubElement(bndbox, 'xmax')  
        xmax.text = str(elem.bbox[1][0])  
        ymax = ET.SubElement(bndbox, 'ymax')  
        ymax.text = str(elem.bbox[1][1])  
        
    # 格式化XML  
    xml_str = ET.tostring(annotation, encoding='unicode')  
    dom = minidom.parseString(xml_str)  
    pretty_xml = dom.toprettyxml(indent="  ")  
    
    return pretty_xml

if __name__ == "__main__":  
    bbox_generator = BBoxGenerator(GroundingDINO(device = 'cuda:6'))
    screenshot_path = '/ssd/xiaxinyuan/code/hmwk/CV_Final_Project/v1/onedrive/frame_1.png'
    output_dir = './output'  
    result = parse_screenshot_layout(screenshot_path, output_dir) 