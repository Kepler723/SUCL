from training import train_no_shuffle, init_parser, \
    make_name, init_model, stay, init_tokenizer
from prepro import read_biored_pubmed
from output import check_times
from utils import read_metadata

import os
import torch

seeds = [28, 46, 77, 89, 94]
args = init_parser().parse_args()
args.attempt_type = [['cl', 'ucl'], ['scl', 'sucl']][args.scl][args.ucl]
args.entities, args.relations, args.alpha, args.beta = read_metadata(args.dataset)
args.exp_attempt = 'ES+EntityMentionAggregation'
args.num_labels = len(args.relations)  
config, tokenizer = init_tokenizer(args)

train_file = os.path.join(args.data_dir, args.dataset, args.train_file)
train_features = read_biored_pubmed(train_file, tokenizer, args)
dev_file = os.path.join(args.data_dir, args.dataset, args.dev_file)
dev_features = read_biored_pubmed(dev_file, tokenizer, args)
test_file = os.path.join(args.data_dir, args.dataset, args.test_file)
test_features = read_biored_pubmed(test_file, tokenizer, args)

for seed in seeds:
    config, tokenizer = init_tokenizer(args)
    args.seed = seed

    model = init_model(args, config, tokenizer)
    if args.load_path != "":
        model.load_state_dict(torch.load(args.load_path, map_location=args.device))

    output_line = make_name(args)
    output_line = '{}_{}'.format(output_line, check_times(output_line))
    args.attempt_name = output_line
    args.save_path = 'result/{}/{}'.format(args.dataset, output_line)
    print(args)
    print('Start training model {}'.format(args.save_path[7:]))

    train_no_shuffle(args, model, train_features, dev_features, test_features)

# stay(args.device)