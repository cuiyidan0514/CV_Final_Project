# CV_Final_Project

环境配置：
方法1：
conda env create -f docker/environment.yml
方法2：
```
conda create -n modelscope python=3.7 # 适配1.15的tensorflow
conda activate modelscope
pip install tensorflow==1.15
pip install modelscope[framework]
pip3 install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

运行文件：
`test.py` 是目前的文件，改`screenshot_file`为需要测试的图片路径，运行即可。

"scrollable”、"clickable”、"selectable”、"disabled"
scrollable: 