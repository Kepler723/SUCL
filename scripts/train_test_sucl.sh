#!/bin/bash


#seeds=(46)

# 循环遍历每个种子值
#for seed in "${seeds[@]}"; do
python all/train_test.py \
  --dataset CDR \
  --model biorex_biolinkbert_pt \
  --train_file train/train_more.tsv \
  --dev_file dev/dev_bioredirect.tsv \
  --test_file test/test_bioredirect.tsv \
  --train_batch_size 14 \
  --test_batch_size 16 \
  --alpha 0.14 \
  --ucl 1 \
  --beta 0.51 \
  --scl 1 \
  --gpuNum 7
#  sleep 5
#done