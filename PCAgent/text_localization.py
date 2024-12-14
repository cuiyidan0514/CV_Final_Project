import cv2
import numpy as np
from PCAgent.crop import crop_image, calculate_size
from PIL import Image


def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points


def longest_common_substring_length(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def ocr(image_path, ocr_detection, ocr_recognition):
    text_data = []
    coordinate = []
    
    image_full = cv2.imread(image_path)
    # 尝试进行文本区域检测，如果检测失败，则返回‘no text’列表和空坐标框
    try:
        det_result = ocr_detection(image_full)
    except:
        print('not text detected')
        return ['no text'], [[0,0,0,0]]
    # print(det_result) # 只是返回多边形框的坐标
    det_result = det_result['polygons'] # 检测到的文本区域是若干个多边形框
    for i in range(det_result.shape[0]): # 对每个多边形框进行遍历
        pts = order_point(det_result[i]) # 将每个框的points按顺时针排列
        image_crop = crop_image(image_full, pts) # 根据points裁剪出包含文本的区域
        
        # 尝试对裁剪出的文本框进行文本识别
        try:
            result = ocr_recognition(image_crop)['text'][0]
        except:
            continue

        # 将bbox的坐标转换为整数，转换为(x1,y1,x2,y2)格式，存入text_data,coordinate列表
        box = [int(e) for e in list(pts.reshape(-1))]
        box = [box[0], box[1], box[4], box[5]]
        
        text_data.append(result)
        coordinate.append(box)

    return text_data, coordinate