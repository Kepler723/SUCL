SUCL: Supervised Contrastive Learning for Biomedical Relation Extraction
项目概述
SUCL（Supervised Contrastive Learning）是一个生物医学关系抽取系统，利用对比学习方法识别和分类生物医学实体之间的关系。该系统处理生物医学文本数据，提取实体（如化学物质、疾病和基因）之间的关系，并应用各种对比学习方法训练模型，以有效识别这些关系。

功能特点
处理多种生物医学数据集（CDR、GDA、DocRed等）
支持多种实体类型（化学物质、疾病、基因等）
实现四种对比学习方法：
基础对比学习 (CL)
无监督对比学习 (UCL)
自对比学习 (SCL)
监督对比学习 (SUCL)
使用BiomedNLP-PubMedBERT作为基础编码器
系统架构
评估

训练方法

元数据系统

数据处理管道

ucl=0, scl=0

ucl=1, scl=0

ucl=0, scl=1

ucl=1, scl=1

生物医学文本

预处理

实体识别

关系抽取

格式化数据 (.tsv)

实体定义

关系类型

训练脚本

基础对比学习 (CL)

无监督对比学习 (UCL)

自对比学习 (SCL)

监督对比学习 (SUCL)

训练模型

测试

性能指标

数据处理流程
SUCL系统包含多个专门处理不同生物医学数据集的脚本：

rel_cdg.py: 处理化学物质-疾病-基因关系 rel_cdg.py:238-242

rel_cdr.py: 专门处理CDR（化学物质-疾病关系）数据集 rel_cdr.py:201-206

rel.py: 通用关系抽取 rel.py:257-262

rel_docred.py: 处理DocRed数据集 rel_docred.py:214-221

rel_gda.py: 处理GDA（基因-疾病关联）数据集 rel_gda.py:166-170

每个处理器执行以下关键步骤：

加载生物医学文本数据
使用SpaCy模型将文本分割成句子
识别文本中的实体及其位置
提取实体间的潜在关系
计算相关实体之间的距离
将输出数据格式化为TSV文件用于训练
支持的实体和关系类型
实体类型
化学物质 (ChemicalEntity)
疾病或表型特征 (DiseaseOrPhenotypicFeature)
基因或基因产物 (GeneOrGeneProduct)
生物体分类 (OrganismTaxon)
序列变异 (SequenceVariant)
细胞系 (CellLine)
关系类型
无关系 (None)
关联 (Association)
正相关 (Positive_Correlation)
结合 (Bind)
负相关 (Negative_Correlation)
比较 (Comparison)
共同治疗 (Cotreatment)
转化 (Conversion)
药物相互作用 (Drug_Interaction)
对比学习方法
SUCL仓库实现了四种主要的对比学习方法，通过ucl和scl参数的组合来选择：

基础对比学习 (CL) (ucl=0, scl=0): 基础方法，学习将相似样本映射到嵌入空间中的相近位置，将不相似样本映射到远离的位置。

无监督对比学习 (UCL) (ucl=1, scl=0): 使用KL散度计算同一输入的两个不同预测分布之间的差异，帮助模型学习更稳健的特征表示。

自对比学习 (SCL) (ucl=0, scl=1): 扩展基础CL，加入自监督组件。使用beta参数控制正则化。

监督对比学习 (SUCL) (ucl=1, scl=1): 结合监督和自监督对比学习。使用alpha和beta参数平衡不同的学习目标。

所有方法都使用生物医学版本的BERT模型，特别是Microsoft的BiomedNLP-PubMedBERT。

模型实现
模型实现中，无监督对比学习通过以下方式实现：

UCL方法使用KL散度来计算两个预测分布之间的差异，并使用alpha参数来控制这个损失的权重：

在命令行参数中，ucl参数用于控制是否启用无监督对比学习：

数据格式和处理
系统使用特殊的标记格式来标识文本中的实体：

源实体标记为 @{EntityType}Src$ 和 @/{EntityType}Src$
目标实体标记为 @{EntityType}Tgt$ 和 @/{EntityType}Tgt$
句子边界标记为 <@sent$>
这种标记过程使模型能够在训练过程中清晰地识别感兴趣的实体及其类型。

支持的数据集
SUCL仓库支持处理和训练多个生物医学数据集：

Bc8: 包含多种生物医学实体类型和关系类型 rel.py:267-269

CDR: 专注于化学物质-疾病关系 rel_cdr.py:211-213

DocRed: 文档级关系抽取数据集 rel_docred.py:227-228

GDA: 基因-疾病关联数据集 rel_gda.py:196-197

chem_dis_gene: 化学物质-疾病-基因关系数据集 rel_cdg.py:262-263

安装与使用
环境要求
Python 3.6+
CUDA支持
SpaCy及其生物医学模型
NLTK
PyTorch
安装步骤
克隆仓库

git clone https://github.com/Kepler723/SUCL.git  
cd SUCL
安装依赖

pip install -r requirements.txt
下载SpaCy模型

python -m spacy download en_core_sci_scibert  
python -m spacy download en_core_sci_lg
数据处理
python process/rel_cdg.py  # 处理化学物质-疾病-基因数据  
python process/rel_cdr.py  # 处理CDR数据  
python process/rel_gda.py  # 处理GDA数据  
python process/rel_docred.py  # 处理DocRed数据  
python process/rel.py  # 处理通用关系数据
模型训练
训练不同的对比学习模型：

# 基础对比学习 (CL)  
python code/train_test.py --ucl 0 --scl 0  
  
# 无监督对比学习 (UCL)  
python code/train_test.py --ucl 1 --scl 0  
  
# 自对比学习 (SCL)  
python code/train_test.py --ucl 0 --scl 1  
  
# 监督对比学习 (SUCL)  
python code/train_test.py --ucl 1 --scl 1
