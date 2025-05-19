# SUCL: Supervised and Unsupervised Contrastive Learning for Document-level Relation Extraction

## 项目概述
SUCL（Supervised and Unsupervised Contrastive Learning）是一个文档级关系抽取框架，利用有监督和无监督对比学习方法识别和分类实体之间的关系。该框架处理关系抽取数据文本，提取并增强实体表征，并应用有监督和无监督对比学习方法有效地区分不同的实体关系。

## 数据集
- 提供处理完毕的数据集，位于<span style="color: gray;">/processed</span>
- 实体和关系的类型信息位于/meta
### 数据格式和处理
使用特殊的标记格式来标识文本中的实体：
- 源实体起首尾位置使用 @{EntityType}Src$ 和 @/{EntityType}Src$标记。
- 目标实体首尾位置使用 @{EntityType}Tgt$ 和 @/{EntityType}Tgt$标记。
- 句子边界使用 <@sent$>标记。
- 没有进行源实体和目标实体的标注，只是为了区分。

## 对比学习
### 有监督学习
按照分类标签，拉近相同标签的实例表征，拉远不同标签的实例表征。
### 无监督学习
使用dropout技术获取两次同一文本表征，拉近表征距离。



## 安装与使用
### 环境要求
- Python 3.7
- PyTorch

### 安装步骤
```bash
git clone https://github.com/Kepler723/SUCL.git  
cd SUCL
pip install -r requirements.txt

# 基础版本 (CL)  
bash scripts/train_test_cl.sh
  
# 无监督对比学习 (UCL)  
bash scripts/train_test_ucl.sh
  
# 有监督对比学习 (SCL)  
bash scripts/train_test_scl.sh
  
# 监督对比学习 (SUCL)  
bash scripts/train_test_sucl.sh
