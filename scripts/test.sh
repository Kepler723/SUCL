#[0:2,1:3,7:5]
python one_cl/test.py --data_dir ../processed \
--transformer_type bert \
--dataset Bc8 \
--model biorex_biolinkbert_pt \
--test_file test_cm_bioredirect.tsv \
--test_batch_size 16 \
--num_labels 9 \
--seed 94 \
--ea 0 \
--es 0 \
--load_path result/Bc8/cl_ea0_es0_seed94_train_dev_more_6_best_test \
--gpuNum 3
