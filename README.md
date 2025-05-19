# SUCL: Supervised and Unsupervised Contrastive Learning for Document-level Relation Extraction

## 项目概述
SUCL（Supervised and Unsupervised Contrastive Learning）是一个文档级关系抽取框架，利用有监督和无监督对比学习方法识别和分类实体之间的关系。该框架处理关系抽取数据文本，提取并增强实体表征，并应用有监督和无监督对比学习方法有效地区分不同的实体关系。

## 数据集
- 提供处理完毕的数据集，位于<span style="color: gray;">/processed</span>
### 数据格式和处理
使用特殊的标记格式来标识文本中的实体：
- 源实体标记为 @{EntityType}Src$ 和 @/{EntityType}Src$
- 目标实体标记为 @{EntityType}Tgt$ 和 @/{EntityType}Tgt$
- 句子边界标记为 <@sent$>

## 系统架构
### 评估
### 训练方法
### 元数据系统
### 数据处理管道

| ucl | scl | 方法 |
| --- | --- | --- |
| 0   | 0   | 基础对比学习 (CL) |
| 1   | 0   | 无监督对比学习 (UCL) |
| 0   | 1   | 自对比学习 (SCL) |
| 1   | 1   | 监督对比学习 (SUCL) |

从生物医学文本到预处理、实体识别、关系抽取、格式化数据 (.tsv) 的过程。

## 生物医学文本处理脚本
- `rel_cdg.py`: 处理化学物质-疾病-基因关系
- `rel_cdr.py`: 专门处理CDR（化学物质-疾病关系）数据集
- `rel.py`: 通用关系抽取
- `rel_docred.py`: 处理DocRed数据集
- `rel_gda.py`: 处理GDA（基因-疾病关联）数据集

每个处理器执行以下关键步骤：加载生物医学文本数据、使用SpaCy模型将文本分割成句子、识别文本中的实体及其位置、提取实体间的潜在关系、计算相关实体之间的距离、将输出数据格式化为TSV文件用于训练。

## 支持的实体和关系类型
### 实体类型
- 化学物质 (ChemicalEntity)
- 疾病或表型特征 (DiseaseOrPhenotypicFeature)
- 基因或基因产物 (GeneOrGeneProduct)
- 生物体分类 (OrganismTaxon)
- 序列变异 (SequenceVariant)
- 细胞系 (CellLine)

### 关系类型
- 无关系 (None)
- 关联 (Association)
- 正相关 (Positive_Correlation)
- 结合 (Bind)
- 负相关 (Negative_Correlation)
- 比较 (Comparison)
- 共同治疗 (Cotreatment)
- 转化 (Conversion)
- 药物相互作用 (Drug_Interaction)

## 对比学习方法
SUCL仓库实现了四种主要的对比学习方法，通过ucl和scl参数的组合来选择。

## 模型实现
所有方法都使用生物医学版本的BERT模型，特别是Microsoft的BiomedNLP-PubMedBERT。

## 数据格式和处理
系统使用特殊的标记格式来标识文本中的实体：
- 源实体标记为 @{EntityType}Src$ 和 @/{EntityType}Src$
- 目标实体标记为 @{EntityType}Tgt$ 和 @/{EntityType}Tgt$
- 句子边界标记为 <@sent$>

## 支持的数据集
- Bc8: 包含多种生物医学实体类型和关系类型
- CDR: 专注于化学物质-疾病关系
- DocRed: 文档级关系抽取数据集
- GDA: 基因-疾病关联数据集
- chem_dis_gene: 化学物质-疾病-基因关系数据集

## 安装与使用
### 环境要求
- Python 3.6+
- CUDA支持
- SpaCy及其生物医学模型
- NLTK
- PyTorch

### 安装步骤
```bash
git clone https://github.com/Kepler723/SUCL.git  
cd SUCL
pip install -r requirements.txt
python -m spacy download en_core_sci_scibert  
python -m spacy download en_core_sci_lg
