# DNABERT2-CMAP

## 环境设置

### 前提条件
- 已安装Anaconda或Miniconda
- 支持CUDA的GPU（推荐用于训练）

### 安装步骤

1. 克隆本仓库
   ```bash
   git clone https://github.com/hualinxu/dnabert2_camp.git
   cd dnabert2_camp
   ```

2. 创建并激活conda环境

   对于Windows用户:
   ```bash
   # 根据提供的YAML文件创建环境
   conda env create -f environment-win.yml
   
   # 激活环境
   conda activate myenv
   ```
   对于Linux用户:
   ```bash
   # 根据提供的YAML文件创建环境
   conda env create -f environment-linux.yml
   
   # 激活环境
   conda activate myenv
   ```

3. （可选）验证环境安装
   ```bash
   conda env list  # 应显示'myenv'为可用环境
   ```

## 代码架构

### 项目目录结构和文件说明
- `data/Ecoli_Promoter_256_independent_test.csv`：独立测试数据集，用于模型验证
- `data/Ecoli_Promoter_8720_balanced.csv`：包含8720条记录的数据集，用作模型训练的基础数据
- `model/.gitattributes`：Git属性配置文件，用于定义Git对文件的处理规则
- `model/bert_layers.py`：用于定义BERT模型的层结构的代码，包括核心网络层的实现
- `model/bert_padding.py`：用于处理BERT模型输入的填充逻辑的代码
- `model/config.json`：模型的核心配置文件，存储模型结构和超参数等关键信息
- `model/configuration_bert.py`：用于定义BERT模型的配置类的代码，用于加载和管理模型配置
- `model/flash_attn_triton.py`：基于Triton实现的Flash Attention（高效注意力机制）的代码
- `model/generation_config.json`：与模型生成任务相关的配置文件，定义生成参数
- `model/LICENSE.txt`：项目许可证文件，说明开源许可条款
- `model/pytorch_model.bin`：PyTorch格式的模型权重文件，存储训练好的模型参数
- `model/README.md`：项目文档，包含项目介绍和使用方法等信息
- `model/tokenizer.json`：Tokenizer的核心配置文件，包含词汇表等信息
- `model/tokenizer_config.json`：Tokenizer的配置参数文件，定义分词规则等
- `dnabert2_camp_train_and_eval.py`：DNABERT2-CAMP 模型的训练和评估脚本
- `promoter_attention_analysis.py`：用于可视化注意力权重和提取显著模体的注意力分析脚本


## 使用方法

### 训练模型

```bash
python `dnabert2_camp_train_and_eval.py`
```

### 分析注意力模式

```bash
python `promoter_attention_analysis.py`
```

### 环境要求
```bash
Python 3.9及以上版本
CUDA支持（可选，用于加速深度学习模型）
```