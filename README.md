## 华为云垃圾分类挑战杯亚军决赛方案分享

### 1.代码结构

```
 {repo_root}
  ├── models	//模型文件夹
  ├── utils		//一些函数包
  |   ├── eval.py		// 求精度
  │   ├── misc.py		// 模型保存，参数初始化，优化函数选择
  │   ├── radam.py
  │   └── ...
  ├── args.py		//参数配置文件
  ├── build_net.py		//搭建模型
  ├── dataset.py		//数据批量加载文件
  ├── preprocess.py		//数据预处理文件，生成坐标标签
  ├── train.py		//训练运行文件
  ├── transform.py		//数据增强文件
```

### 2. 环境设置

可以直接通过`pip install -r requirements.txt`安装指定的函数包，python版本为3.6，具体的函数包如下：

* pytorch=1.0.1
* torchvision==0.2.2
* matplotlib==3.1.0
* numpy==1.16.4
* scikit-image
* pandas
* sklearn

### 3.方案思路

[方案讲解](https://mp.weixin.qq.com/s/7GhXMXQkBgH_JVcKMjCejQ)

知乎专栏：[ML与DL成长之路](https://zhuanlan.zhihu.com/c_1138029910563020800