from training import train_no_shuffle, report, evaluate, init_parser, set_seed, make_name, init_tokenizer,init_model,metric_table,rel_id
from prepro import read_biored_pubmed
import os
import torch
import tqdm
from output import output, format_test, check_times


args = init_parser().parse_args()

config, tokenizer = init_tokenizer(args)


print(args)

model = init_model(args, config, tokenizer)


# train_file = os.path.join(args.data_dir, args.dataset, args.train_file)
# train_features = read_biored_pubmed(train_file, tokenizer, args.num_labels)
# dev_file = os.path.join(args.data_dir, args.dataset,args.dev_file)
# dev_features = read_biored_pubmed(dev_file, tokenizer, 9)
test_file = os.path.join(args.data_dir, args.dataset,args.test_file)
test_features = read_biored_pubmed(test_file, tokenizer)


# if args.dataset != 'BioRED1':
#     test_features = read_biored_pubmed(test_file, tokenizer, 9)

if args.load_path != "":
    model.load_state_dict(torch.load(args.load_path, map_location=args.device))

output_line = make_name(args)
output_line = '{}_{}'.format(output_line, check_times(output_line))
args.attempt_name = output_line
args.save_path = 'result/{}/{}'.format(args.dataset, output_line)

print('Start test model {}'.format(args.save_path[7:]))

test_score, test_output=evaluate(args, model, test_features)
print(metric_table(test_output, 'test',9))



# if args.dataset == 'BioRED1':
#     test_features = read_biored_pubmed(test_file, tokenizer, max_seq_length=args.max_seq_length)
#     preds = report(args, model, test_features)
#     model.load_state_dict(torch.load(args.save_path, map_location=device))
#     print('Start testing on model {}'.format(args.load_path[7:]))
#     output_line = '{}_{}'.format(output_line, args.test_file[5:][:-4])
#     store_preds(output_line, preds, test_file)
# stay(device)


