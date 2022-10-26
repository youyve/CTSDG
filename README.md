# 目录

- [目录](#目录)
- [1. CTSDG 模型介绍](#1-ctsdg-模型介绍)
  - [1.1. 网络模型结构](#11-网络模型结构)
  - [1.2. 数据集](#12-数据集)
  - [1.3. 代码提交地址](#13-代码提交地址)
  - [1.4. 其它](#14-其它)
- [2. 代码目录结构说明](#2-代码目录结构说明)
  - [2.1. 脚本参数](#21-脚本参数)
- [3. 自验结果](#3-自验结果)
  - [3.1. 自验环境](#31-自验环境)
  - [3.2. 训练超参数](#32-训练超参数)
  - [3.3. 训练](#33-训练)
    - [3.3.1. 训练之前](#331-训练之前)
    - [3.3.2. 启动训练脚本](#332-启动训练脚本)
  - [3.4. 评估过程](#34-评估过程)
    - [3.4.1. 启动评估脚本](#341-启动评估脚本)
    - [3.4.2. 评估精度结果](#342-评估精度结果)
  - [4.1. 参考论文](#41-参考论文)
  - [4.2. 参考git项目](#42-参考git项目)

# [1. CTSDG 模型介绍](#contents)

深度生成方法最近通过引入结构先验在图像修复方面取得了长足的进步。然而，由于在结构重建过程中缺乏与图像纹理的适当交互，目前的解决方案在处理大腐败的情况时能力不足，并且通常会导致结果失真。CTSDG 是一种新颖的用于图像修复的双流网络，它以耦合的方式对结构约束的纹理合成和纹理边缘引导结构重建进行建模，使它们更好地相互利用，以获得更合理的生成。此外，为了增强全局一致性，设计了双向门控特征融合( Bi-GFF )模块来交换和结合结构和纹理信息，并开发了上下文特征聚合( CFA )模块，通过区域亲和学习和多尺度特征聚合来细化生成的内容。

## [1.1. 网络模型结构](#contents)

## [1.2. 数据集](#contents)

用到的数据集: 

[CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 

[NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

- 需要从 **CELEBA** 下载以下内容：

  - `img_align_celeba.zip`
  - `list_eval_partitions.txt`

- 需要从 **NVIDIA Irregular Mask Dataset** 下载以下内容：

  - `irregular_mask.zip`
  - `test_mask.zip`

- 目录结构如下：

  ```text
    .
    ├── img_align_celeba            # 图像文件夹
    ├── irregular_mask              # 用于训练的遮罩
    │   └── disocclusion_img_mask
    ├── mask                        # 用于测试的遮罩
    │   └── testing_mask_dataset
    └── list_eval_partition.txt     # 拆分文件
  ```

## [1.3. 代码提交地址](contents)

https://git.openi.org.cn/youlz/CTSDG

# [2. 代码目录结构说明](#contents)

```text
.
├── converter.py                 # 将 VGG16 转换为 mindspore 的 checkpoint
├── dataset
│   ├── img_align_celeba         # celeba 图像文件夹
│   ├── irregular_mask           # 用于训练的遮罩
│   ├── list_eval_partition.txt  # 拆分文件
│   └── mask                     # 用于测试的遮罩
├── default_config.yaml          # 默认配置文件
├── eval.py                      # 评估 mindspore 模型
├── __init__.py                  # 初始化文件
├── model_utils
│   ├── config.py                # 语法参数
│   └── __init__.py              # 初始化文件
├── requirements.txt
├── scripts
│   ├── run_eval_npu.sh          # 在 NPU 上启动评估的脚本
│   └── run_train_npu.sh         # 在 NPU 上启动训练的脚本
├── src
│   ├── callbacks.py             # 回调
│   ├── dataset.py               # celeba 数据集
│   ├── discriminator            # 鉴别器
│   ├── generator                # 生成器
│   ├── initializer.py           # 初始化器权重
│   ├── __init__.py              # 初始化文件
│   ├── losses.py                # 模型 loss
│   ├── trainer.py               # ctsdg模型的训练者
│   └── utils.py                 # 工具
├── train.py                     # 训练 mindspore 模型
└── vgg16-397923af.pth           # VGG16 torch 模型
```

## [2.1. 脚本参数](#contents)

可以在 `default_config.yaml` 中配置训练参数

```text
"gen_lr_train": 0.0002,                     # 生成器训练的 lr
"gen_lr_finetune": 0.00005,                 # 生成器微调的 lr
"dis_lr_multiplier": 0.1,                   # 判别器的 lr 是生成器的 lr 乘以这个参数
"batch_size": 6,                            # batch size
"train_iter": 350000,                       # 训练迭代次数
"finetune_iter": 150000                     # 微调迭代次数
"image_load_size": [256, 256]               # 输入图像大小
```

有关更多参数，请参见 `default_config.yaml` 的内容。

# [3. 自验结果](contents)

## [3.1. 自验环境](contents)

- 硬件环境
  - CPU：aarch64  192核 
  - NPU：910ProA 32G
- MindSpore version:  1.6.1
- Python
  - 版本：Python 3.7.6
  - 第三方库和依赖：requirements.txt

## [3.2. 训练超参数](contents)

train_iter : 350000

finetune_iter : 150000 

gen_lr_train : 0.0002

gen_lr_finetune : 0.00005 

dis_lr_multiplier : 0.1 

batch_size : 6

Loss function : GWithLossCell() , DWithLossCell()

Optimizer : Adam

## [3.3. 训练](contents)

### [3.3.1. 训练之前](#contents)

对于训练 CTSDG 模型，需要对 VGG16 torch 模型进行感知损失转换。

1. [下载预训练的 VGG16](https://download.pytorch.org/models/vgg16-397923af.pth)
2. 将 torch checkpoint 转换为 mindspore :

```shell
python converter.py --torch_pretrained_vgg=/path/to/torch_pretrained_vgg
```

转换后的 mindpore checkpoint 将保存在与 torch 模型相同的目录中，名称为`vgg16_feat_extr_ms.ckpt`。

After preparing the dataset and converting VGG16 you can start training and evaluation as follows：

准备好数据集同时完成 VGG16 的转换后，就可以通过如下步骤开始训练和评估模型了。

### [3.3.2. 启动训练脚本](contents)

```shell
# train
bash scripts/run_train_npu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

Example:

```shell
# DEVICE_ID - 用于训练的设备 ID 号
# CFG_PATH - config 的路径
# SAVE_PATH - 保留 logs and checkpoints 的路径
# VGG_PRETRAIN - 预训练 VGG16 的路径
# IMAGES_PATH - CELEBA 数据集的路径
# MASKS_PATH - 用于训练的遮罩路径
# ANNO_PATH - 拆分文件的路径
bash scripts/run_train_npu.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

## [3.4. 评估过程](#contents)

### [3.4.1. 启动评估脚本](contents)

```shell
# evaluate
bash scripts/run_eval_npu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

Example:

```shell
# evaluate
# DEVICE_ID - 用于评估的设备 ID 号
# CFG_PATH - config 的路径
# CKPT_PATH - path to ckpt for evaluation用于评估的 ckpt 的路径
# IMAGES_PATH - CELEBA 数据集的路径
# MASKS_PATH - 用于测试的遮罩路径
# ANNO_PATH - 拆分文件的路径
bash scripts/run_eval_npu.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt  
```

## [4.1. 参考论文](contents)

- [Image Inpainting via Conditional Texture and Structure Dual Generation](https://arxiv.org/pdf/2108.09760.pdf)

## [4.2. 参考git项目](contents)

- https://github.com/Xiefan-Guo/CTSDG

- https://github.com/mindspore-ai/models/tree/master/research/cv/CTSDG
