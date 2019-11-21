## 华为云垃圾分类挑战杯亚军方案分享

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

* pytorch>=1.0.1
* torchvision==0.2.2
* matplotlib>=3.1.0
* numpy>=1.16.4
* scikit-image
* pandas
* sklearn

注：py3.7训练的话，要修改下面的代码
`if use_cuda: inputs, targets = inputs.cuda(), targets.cuda(async=True) inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)`
\#python3.7已经移除了async关键字，而用non_blocking代替。(导致apache-airflow也出了问题)
\#cuda() 本身也没有async.

就是把 async=True去掉

if use_cuda:
inputs, targets = inputs.cuda(), targets.cuda()
inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)`

## 3.运行步骤

1. 建立文件夹data，把garbage_classify全部解压缩到data下
2. 运行preprocess.py，生成训练集和测试集运行文
3. 单张显卡的话，修改arg.py 85行 parser.add_argument('--gpu-id', default='0, 1, 2, 3' 为'--gpu-id', default='0'，同时修改 '--train-batch'，'--test-batch'为适当的数字
4. 运行train.py

### 4.方案思路

[方案讲解](https://mp.weixin.qq.com/s/7GhXMXQkBgH_JVcKMjCejQ)

知乎专栏：[ML与DL成长之路](https://zhuanlan.zhihu.com/ai-growth)

如果复现过程中有bug，麻烦反馈一下，会优化更新。如果对您有帮助记得给个**star**

---

**小尾巴**

QQ群：AI成长社①：545702197

微信群：添加微信号：Derek_wen8，备注：加群