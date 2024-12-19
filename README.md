# CV_Final_Project

全局变量：
切换数据集时，修改411-415的全局变量，并自行创建对应的文件夹
在418行设置在该数据集中希望检测的图片，eg：frame_1, frame_2_1

超参数：426-432
caption：要检索的关键词，建议为单个名词，eg：checkbox, radio buttons, icons
sub_num: 进行det时图片的分割粒度，可选范围{4,9,16,25,36}，一般4或9即可，数字越大检测到的越精细，但假阳样本也会增多
bbox_th: 只有det的预测置信度大于bbox_th时才会将bbox标注，可以根据需要调整
text_only: 是否允许单独的文本被标注
icon_only：是否允许单独的icon被标注
'''解释一下，根据大部分app截图的特性，每个button边上都会有文本说明，所以一般认为只有icon附近有text的时候，才认为这确实是一个组件，而不是噪声'''
clear_ocr: 是否要把ocr检测到的文本给清除后再进行检测，有时文本会造成干扰，但有时文本是必须的(比如超链接)
load_ocr: 是否加载已经生成的ocr文本框坐标，只有在每张图第一次处理时设置为false，会花费一些时间，后面再在该图上检测时直接加载已生成的ocr即可

注释相关：
443：如果希望绘制标准format，即{'clickable','selectable','scrollable','format'}则取消该行注释
447-469：一次输入多个提示词，将获得的结果进行merge，如果同一个框被多个关键词同时检测到，则取最高的置信度对应的标签作为该bbox的label
373：只根据检测到的icon进行绘制并保存
308，335-338：如果希望自行检测root，就取消注释
307，332-333：如果希望根据xml直接裁剪root，就取消注释

重要文件说明：
run_original.py：多个图片分割函数，其中sub4分割实际上分割成了9份，彼此之间有重叠
PCAgent.merge_strategy：多个不同的合并策略，根据不同需求设计了不同的merge方式
detect_root_area.py：根据文件名检索上一张图片，并用传统cv方法分割出root并保存图片
eval_main.py：计算ap

./dataset/train_root_dataset:保存切割后的root区域、ocr检测结果以及挖去ocr部分的结果
./screenshot/jiguang:保存检测结果
./output/jiguang:保存每张图片检测后生成的csv文件，先分开存放，方便观察每张图片的ap，后续可将一个数据集的所有csv合并为一个文件，再进行ap检测
./eval_output:保存执行eval_main.py后生成的metric

运行文件：
`test.py` 是目前的文件，改`screenshot_file`为需要测试的图片路径，运行即可。

# train using dino prompt
1. 主程序为inference.py, run.sh 为其运行文件（需要修改注释）。修改 mode = 'my_train' 即可，其会保存 `hs` 和IoU大于0.2的gt label，用于可能的后续测试为`pkl`文件
2. 训练代码在 `models/train_dino.py` (未完成)

# train using rol pooling
1. 修改`models/train_rol.py`中的`train_model`函数，设置`resume=True`，并指定`checkpoint_path`为之前训练好的模型路径
2. 运行run.sh, 其中的参数设置是为了指定GPU
3. tensorboard 保存在runs/下, ckpt保存至主目录下