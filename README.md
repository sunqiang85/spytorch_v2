# 深度学习项目说明
## 数据分类
数据分为：训练数据train，验证数据validation和测试数据test。其中test数据集的label可能有，也可能没有（challenge）。

## 任务
任务可分为：训练，测试
- 训练：在训练数据集上训练，在验证集上验证
- 测试：在训练数据集和验证集上训练，在测试集上验证

其中训练可以从零开始，也可以是Finetune，因此总体而言训练任务可分为以下3种情况

| 任务类型 | 加载 | 训练集        | 验证集  |
| -------- | -------- | ---------------- | ---------- |
| 从零训练 | 无      | train            | validation |
| 继续训练 | 模型，优化器，engine状态      | train            | validation |
| Finetune | 模型      | train            | validation |
| 测试   | 模型      | train+validation | test       |

因此一个engine必须包换三个方法
- train
- validate
- load_state (模型，优化器，engine状态）

从加载的角度，任务可以分为三种状态
- scratch: 从零开始训练
— continue: 继续训练
- pretrained: Finetune, 测试


## 配置文件
配置文件中的配置可以分为两种类型
- 自生变量，可以直接定义如data_root
- 衍生变量，可以利用property产生，如训练数据路径data_root/train.pt

配置类需要支持以下功能：
- 从yaml中更新
- 从dict中更新（argparse解析从dict）

## 日志
- 更新文本日志
- 更新指标：loss，accuracy
- 更新图片：tensorboardx中生成的图


## 多数据集
如果有多个数据集的话，必须支持统一接口，所以通过字典的方式返回是一个较好的方式，可以按需取数据。一个好的数据集支持以下特性
- 返回字典
- 支持限定类别的种类
- 支持限定每个类别的数量（分训练和验证）


## 多任务
每个任务对应了：模型，优化器和损失函数


## 训练
```bash
# from scratch
python main.py

# from pretrained, modify the pretrained_ckpt_path in yaml
python main.py --init pretrained

# from previous trained model, modify the resume_ckpt_path in yaml
python main.py --init resume

```

## 测试
```bash
python main.py --mode 'test'
```


