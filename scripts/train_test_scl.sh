#!/bin/bash

# 定义种子值列表

    # 运行Python脚本进行训练和测试
python all/train_test.py \
    --dataset CDR \
    --model biorex_biolinkbert_pt \
    --train_file train/train_more.tsv \
    --dev_file dev/dev_bioredirect.tsv \
    --test_file test/test_bioredirect.tsv \
    --train_batch_size 16 \
    --test_batch_size 16 \
    --scl 1 \
    --gpuNum 3

# 等待5秒
sleep 5

