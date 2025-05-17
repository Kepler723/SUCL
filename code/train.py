from training import train, init_parser, set_seed, make_name, init_model, stay
import os
import torch
from output import check_times
from  prepro import read_biored_pubmed


args = init_parser().parse_args()
print(args)
device, config, tokenizer, model = init_model(args)

train_file = os.path.join(args.data_dir, args.train_file)
train_features = read_biored_pubmed(train_file, tokenizer, max_seq_length=args.max_seq_length)
if args.load_path != "":
    model.load_state_dict(torch.load(args.load_path, map_location=device))
output_line = make_name(args)
output_line = '{}_{}'.format(output_line, check_times(output_line))
args.save_path = 'result/{}'.format(output_line)
print('Start training model {}'.format(args.save_path[7:]))
train(args, model, train_features)

stay(device)
