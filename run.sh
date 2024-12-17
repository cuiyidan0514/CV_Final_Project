#!/bin/bash
# 定义路径变量
mode="baseline"

IOU_THRESH=0.3
ERR_S=0.3
# 数据集路径
IMG_ROOT="v1/skype"
XML_ROOT="v1/skype"
# 数据集inference输出
OUTPUT_ROOT="output/v1/skype"
# 错误可视化
OTT_IMG_ROOT="eval_wrong/skype/"
# 算法输出结果CSV
CSV_ROOT="output/v1/skype/csv"
# 评估输出
EVAL_OUTPUT="./eval_output/skype/"
# 评估类别
CLASS_NAME=("clickable" "scrollable" "selectable" "disabled")


# python inference.py \
#     --mode $mode \
#     --input_dir $IMG_ROOT \
#     --output_dir $OUTPUT_ROOT \
#     --csv_output_dir $CSV_ROOT


# 运行 eval_main.py 并传递参数
python eval_main.py \
    --iou_thresh $IOU_THRESH \
    --err_s $ERR_S \
    --img_root "$IMG_ROOT" \
    --xml_root "$XML_ROOT" \
    --ott_img_root "$OTT_IMG_ROOT" \
    --csv_root "$CSV_ROOT" \
    --eval_output "$EVAL_OUTPUT" \
    --class_name "$CLASS_NAME"