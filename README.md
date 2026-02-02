# 花卉识别项目

基于深度学习的花卉图像分类项目，使用PyTorch实现。包含训练、预测和交互式Web测试界面。

## 项目结构

```
Flower/
├── data/                  # 数据集目录
│   └── flowers/          # 花卉图片数据集
│       ├── train/         # 训练集（已按类别分类）
│       │   ├── daisy/    # 雏菊
│       │   ├── dandelion/# 蒲公英
│       │   ├── rose/     # 玫瑰
│       │   ├── sunflower/# 向日葵
│       │   └── tulip/    # 郁金香
│       └── test/         # 测试集（未标注）
├── models/               # 模型定义
│   └── model.py         # 模型架构
├── utils/                # 工具函数
│   └── data_loader.py    # 数据加载器
├── checkpoints/          # 模型检查点保存目录
│   ├── best_model.pth   # 最佳模型
│   └── logs/           # TensorBoard日志
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── app.py               # Web应用后端
├── index.html           # Web应用前端
├── config.json           # 配置文件
└── requirements.txt      # 依赖包
```

## 功能特性

- 🎯 **模型训练**: 支持多种预训练模型（ResNet18/50、EfficientNet、VGG16）
- 📊 **数据增强**: 自动应用随机翻转、旋转、颜色抖动等增强技术
- 📈 **训练监控**: 集成TensorBoard，实时查看训练曲线
- 🔮 **模型预测**: 支持单张和批量图片预测
- 🌐 **Web测试界面**: 交互式网页，手动测试模型准确率
- 💾 **模型保存**: 自动保存最佳模型和检查点

## 环境配置

### 系统要求

- **Python版本**: 3.9 或更高版本
- **操作系统**: Windows, Linux, macOS
- **内存**: 建议 8GB 以上 RAM
- **GPU**: 可选
  - 默认使用 CPU 训练
  - 如需使用 GPU 加速，需要修改代码中的设备设置
  - GPU 训练速度通常比 CPU 快 10-50 倍

### 1. 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create -n flower python=3.9
conda activate flower

# 或使用venv创建虚拟环境
python -m venv venv
# Windows激活
venv\Scripts\activate
# Linux/Mac激活
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集准备

### 当前数据集结构

项目已配置为使用Kaggle花卉识别数据集，数据集结构如下：

```
data/flowers/
├── train/              # 训练集（已按类别分类）
│   ├── daisy/          # 雏菊
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── dandelion/      # 蒲公英
│   │   ├── image1.jpg
│   │   └── ...
│   ├── rose/           # 玫瑰
│   │   ├── image1.jpg
│   │   └── ...
│   ├── sunflower/      # 向日葵
│   │   └── ...
│   └── tulip/          # 郁金香
│       └── ...
├── test/               # 测试集（未标注，用于Web测试）
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── ...
├── Testing_set_flower.csv
└── sample_submission.csv
```

### 数据集说明

- **训练集**: 包含5个花卉类别，每个类别约500-650张图片，共2746张
- **测试集**: 未标注的测试图片，共924张，用于Web测试界面
- **数据划分**: 训练集会自动划分为训练集(80%)、验证集(10%)、测试集(10%)

### 方式一：使用Kaggle数据集

1. 访问 [Kaggle Flower Recognition](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/data)
2. 下载数据集并解压
3. 将解压后的文件放入 `data/flowers/` 目录

### 方式二：使用自定义数据集

1. 在 `data/flowers/train/` 目录下创建花卉类别的子文件夹
2. 将对应类别的图片放入相应的文件夹中
3. 支持的图片格式：.jpg, .jpeg, .png, .bmp, .gif
4. 每个类别至少需要100张图片以获得较好的训练效果

## 训练模型

### 基础训练

```bash
python train.py
```

### 自定义配置训练

编辑 [train.py](train.py) 中的配置参数：

```python
config = {
    'data_dir': 'data/flowers',      # 数据集目录
    'batch_size': 16,                # 批次大小
    'image_size': 224,               # 图片尺寸
    'num_workers': 0,                # 数据加载线程数
    'model_type': 'resnet18',        # 模型类型: resnet18, resnet50, efficientnet, vgg16
    'pretrained': True,              # 是否使用预训练权重
    'learning_rate': 0.001,          # 学习率
    'weight_decay': 1e-4,           # 权重衰减
    'epochs': 30,                    # 训练轮数
    'checkpoint_dir': 'checkpoints'  # 检查点保存目录
}
```

**注意**: 
- 默认使用 CPU 训练
- 如需使用 GPU，请修改代码中的 `self.device = torch.device('cpu')` 为 `torch.device('cuda')`
- GPU 训练速度通常比 CPU 快 10-50 倍

### 查看训练过程

训练过程中会自动生成TensorBoard日志，可以使用以下命令查看：

```bash
tensorboard --logdir checkpoints/logs
```

然后在浏览器中打开 `http://localhost:6006`

## 模型预测

### 单张图片预测

```bash
python predict.py --image path/to/flower.jpg --checkpoint checkpoints/best_model.pth
```

### 批量预测

```bash
python predict.py --image_dir path/to/images/ --checkpoint checkpoints/best_model.pth
```

### 预测并可视化

```bash
python predict.py --image path/to/flower.jpg --visualize --save_dir predictions
```

### 预测参数说明

- `--checkpoint`: 模型检查点路径（默认: checkpoints/best_model.pth）
- `--image`: 单张图片路径
- `--image_dir`: 图片目录路径（批量预测）
- `--top_k`: 显示前K个预测结果（默认: 5）
- `--visualize`: 是否可视化预测结果
- `--save_dir`: 可视化结果保存目录（默认: predictions）

## Web测试界面

### 启动Web服务

```bash
python app.py
```

启动后在浏览器中访问：http://localhost:5000

### 功能说明

Web测试界面提供以下功能：

- 🎲 **随机图片**: 从test目录随机选择图片
- 🤖 **模型预测**: 自动显示模型预测结果和置信度
- ✋ **手动标注**: 你手动判断图片的正确类别
- 📊 **实时统计**: 实时显示轮次、正确数量和准确率
- 🎯 **20轮测试**: 完成20轮后显示最终准确率
- 🔄 **重置功能**: 随时重置测试，开始新一轮

### 使用流程

1. 点击"下一张图片"按钮
2. 查看模型预测结果（中文显示）
3. 根据图片内容手动选择正确的类别
4. 系统自动对比你的选择和模型预测
5. 显示正确/错误反馈并更新统计
6. 重复20轮后显示最终结果

### 类别说明

| 编号 | 英文 | 中文 | 特征 |
|------|------|------|------|
| 0 | daisy | 雏菊 | 白色花瓣，黄色花心，花瓣细长 |
| 1 | dandelion | 蒲公英 | 黄色花朵，绒球状，有白色绒毛 |
| 2 | rose | 玫瑰 | 多层花瓣，颜色多样（红、粉、白等）|
| 3 | sunflower | 向日葵 | 大型黄色花朵，棕色花心，花瓣宽大 |
| 4 | tulip | 郁金香 | 杯状花朵，颜色多样，花瓣较少 |

## 模型选择

项目支持多种预训练模型，可通过修改 `model_type` 参数选择：

- **resnet18**: ResNet-18，轻量级，训练快速（推荐CPU训练）
- **resnet50**: ResNet-50，平衡性能和速度
- **efficientnet**: EfficientNet-B0，高效模型
- **vgg16**: VGG-16，经典模型

## 训练结果

本项目在Kaggle花卉数据集上的训练结果：

| 模型 | 测试准确率 | 训练轮数 | 训练时间 |
|------|-----------|----------|----------|
| ResNet18 | 92.36% | 25 | ~1小时（CPU）|
| ResNet50 | ~94% | 30 | ~2小时（CPU）|

### 各类别表现

| 花卉类别 | 准确率 | 精确率 | 召回率 | F1分数 |
|---------|--------|--------|--------|---------|
| 向日葵 | 96.97% | 96.00% | 96.97% | 96.48% |
| 蒲公英 | 95.35% | 95.35% | 95.35% | 94.98% |
| 雏菊 | 93.00% | 92.38% | 93.00% | 92.66% |
| 玫瑰 | 90.00% | 83.33% | 90.00% | 86.54% |
| 郁金香 | 86.89% | 92.17% | 86.89% | 89.45% |

## 训练策略

1. **数据增强**: 使用随机翻转、旋转、颜色抖动等增强技术
2. **预训练权重**: 默认使用ImageNet预训练权重加速收敛
3. **学习率调度**: 使用余弦退火学习率调度器
4. **早停机制**: 自动保存验证集上表现最好的模型
5. **优化器**: 使用AdamW优化器，配合权重衰减

## 常见问题

### 1. CUDA out of memory

减小 `batch_size` 或使用更小的模型（如resnet18）

### 2. 数据集加载错误

检查数据集目录结构是否正确，确保每个类别文件夹都包含图片

### 3. 训练速度慢

- 增加 `num_workers`（建议设置为CPU核心数）
- 使用GPU训练
- 减小 `image_size`
- 使用更轻量的模型（如ResNet18）

### 4. 模型不收敛

- 降低 `learning_rate`
- 增加 `epochs`
- 检查数据集质量和类别平衡
- 尝试不同的数据增强策略

### 5. Web服务无法启动

- 确保已安装Flask和Flask-CORS: `pip install flask flask-cors`
- 检查端口5000是否被占用
- 查看防火墙设置

## 性能优化建议

1. **数据集**: 每个类别至少100-200张图片
2. **训练轮数**: 根据数据集大小调整，通常20-50轮
3. **学习率**: 从0.001开始，可根据训练情况调整
4. **批次大小**: 根据GPU内存调整，建议16-64
5. **模型选择**: CPU训练建议使用ResNet18，GPU训练可使用ResNet50

## 扩展功能

### 添加新的花卉类别

只需在 `data/flowers/train/` 目录下添加新的类别文件夹，并将图片放入其中即可。

### 微调预训练模型

修改 [train.py](train.py) 中的配置，设置 `pretrained=True` 即可使用预训练权重。

### 导出模型为ONNX格式

```python
import torch

from models.model import create_model

model = create_model(num_classes=5, model_type='resnet18', pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'flower_model.onnx')
```

## 技术栈

- **深度学习框架**: PyTorch 2.1.0
- **计算机视觉**: TorchVision
- **数据处理**: NumPy, Pillow
- **可视化**: Matplotlib
- **训练监控**: TensorBoard
- **Web框架**: Flask + Flask-CORS
- **前端**: HTML5 + CSS3 + JavaScript

## 许可证

本项目仅供学习和研究使用。

## 致谢

- 数据集来源: [Kaggle Flower Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- 预训练模型: PyTorch TorchVision

## 联系方式

如有问题或建议，欢迎提出Issue。
