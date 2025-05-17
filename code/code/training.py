from torch.utils.data import DataLoader, Sampler
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn_pubmed_withEntityLocation
from tabulate import tabulate
from bidict import bidict
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import tqdm
import time
import argparse
import os
import random
import numpy as np
import torch
import ujson as json


loss_rates = [1]
scores = [[],[]]

epoch_loss = []
# os.environ["WANDB_API_KEY"] = "d893ae34e52f1394807f7fa54cc79de1b0d000ad"
# os.environ["WANDB_MODE"] = "offline"
def custom_round(value, decimals=2):
    multiplier = 10 ** decimals
    return int(value * multiplier + 0.5) / multiplier

def train_no_shuffle(args, model, train_features, dev_features, test_features=None):
    # custom_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_temp")
    # os.makedirs(custom_dir, exist_ok=True)
    # os.environ["WANDB_DIR"] = custom_dir
    # print(custom_dir)
#     wandb.init(
#         project="Bc8",
#         name=args.attempt_name,
# )
    def finetune(features, optimizer, num_epoch, num_steps):
        best_loss = 500
        best_dev_epoch = 0
        best_test_epoch = 0
        best_test = -1
        best_dev = -1
        best_dev_test = -1

        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True,
                                      collate_fn=collate_fn_pubmed_withEntityLocation, drop_last=False)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        scaler = GradScaler()
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epochNum, epoch in enumerate(train_iterator):
            print("---------------------------------------------------------------")
            print("当前正在训练第" + str(epochNum+1) + "个epoch!")
            model.zero_grad()
            for step, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'entity_pos': batch[2],
                          'sent_pos': batch[3],
                          'labels': batch[4],
                          'src_type': batch[5],
                          'tgt_type': batch[6],
                          'dist': batch[7],
                          }

                with autocast():
                    outputs = model(args, **inputs)
                    loss = outputs / args.gradient_accumulation_steps

                    if args.dataset == 'BioRED1':
                        temp_loss = loss.item()
                        for loss_rate in loss_rates:
                            temp_loss *= loss_rate
                        if temp_loss <= best_loss:
                            best_loss = temp_loss
                            if best_loss < 1e-5:
                                loss_rates[-1] *= 100
                                if loss_rates[-1] == 10000000000:
                                    loss_rates.append(1)
                                best_loss *= 100
                            # if best_loss < 1e-3:
                            #     torch.save(model.state_dict(), '{}_best'.format(args.save_path))
                # torch 版本
                scaler.scale(loss).backward()
                # 普通版本
                # loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # optimizer.step()
                    # scheduler.step()
                    scaler.step(optimizer)
                    # scaler.step(scheduler)

                    scaler.update()
                    scheduler.step()

                    model.zero_grad()
                    num_steps += 1
                # scaler.update()
            # log_model_parameters(model, epochNum)

            # wandb.log({"loss": loss.item()})
            # wandb.log({"alpha": args.alpha})
            # wandb.log({"beta": args.beta})
            print("Training Loss:" + str(loss.item()))
            print("---------------------------------------------------------------")
            dev_score, dev_output = evaluate(args, model, dev_features, "dev", epochNum)
            test_score, test_output = evaluate(args, model, test_features, "test", epochNum)
            # for item, value in dev_output.items():
            #     wandb.log({item: value}, step=num_steps)
            # for item, value in test_output.items():
            #     wandb.log({item: value}, step=num_steps)
            # wandb.log(dev_score, step=epochNum)
            # wandb.log(test_score, step=epochNum)
            if dev_score["micro_f1"] > best_dev:
                best_dev = dev_score["micro_f1"]
                best_dev_test = test_score["micro_f1"]
                best_dev_epoch = epochNum + 1
                torch.save(model.state_dict(), '{}_best_dev'.format(args.save_path))

            if test_score["micro_f1"] > best_test:
                best_test = test_score["micro_f1"]
                best_test_epoch = epochNum+1
                torch.save(model.state_dict(), '{}_best_test'.format(args.save_path))

            write_metric(args, dev_output, "dev", best_dev, best_dev_epoch, epochNum)
            write_metric(args, test_output, "test", best_test, best_test_epoch, epochNum)

            print('best test score {} based on dev score at epoch {} '.format(best_dev_test, best_dev_epoch))
            print('best test score {} at epoch {}'.format(best_test, best_test_epoch))
            print("---------------------------------------------------------------")

    new_layer = ["extractor", "bilinear", "contrastiveLoss", "cl_linear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)

def write_metric(args, metric, tag, best_score, best_epoch, epochNum):
    print('Epoch: {}    {}'.format(epochNum + 1, tag))
    print(metric_table(args.relations, metric, tag, args.num_labels))
    with open('../metric/{}/{}/{}/{}.txt'.format(args.dataset, args.model, args.attempt_type, '{}_{}'.format(tag, args.attempt_name)),
              'a') as file:
        file.write('Epoch: {}\n'.format(epochNum + 1))
        file.write(metric_table(args.relations, metric, tag, args.num_labels))
        if tag == "dev":
            file.write('best test score {} based on dev score at epoch {}\n'.format(best_score, best_epoch))
        else:
            file.write('test score {} at epoch {}\n'.format(best_score, best_epoch))
        file.write('---------------------------------------------------------------\n')

def evaluate(args, model, features, tag="test", epoch=26):
    id2rel = args.relations.inverse

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False,
                            collate_fn=collate_fn_pubmed_withEntityLocation, drop_last=False)
    preds, golds = [], []
    if not os.path.exists('../error/{}/{}/{}/{}/'.format(args.dataset, args.model, args.attempt_type, args.attempt_name)):
        os.makedirs('../error/{}/{}/{}/{}/'.format(args.dataset, args.model, args.attempt_type, args.attempt_name))
    with open('../error/{}/{}/{}/{}/{}_{}.txt'.format(args.dataset, args.model, args.attempt_type, args.attempt_name, tag, epoch), 'w') as file:

        for batch in tqdm.tqdm(dataloader,total=len(dataloader)):
            model.eval()

            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'entity_pos': batch[2],
                      'sent_pos': batch[4],
                      'src_type': batch[5],
                      'tgt_type': batch[6],
                      'dist': batch[7],
                      }

            with torch.no_grad():
                # pred, *_ = model(**inputs)
                pred = model(args, **inputs)
                pred = pred.cpu().numpy()
                pred[np.isnan(pred)] = 0
                preds.append(pred)
                # golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))
                golds.append(np.array(batch[4], np.float32))

                for index in range(len(pred)):
                    p_num = np.argmax(pred[index])
                    g_num = batch[4][index].index(1)

                    if p_num != g_num:
                        file.write(str(batch[9][index])+'\t'+batch[8][index].strip('\n')+'\t{}\t{}\n'.format(id2rel[g_num],id2rel[p_num]))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)

    f1 = {}
    p = {}
    r = {}
    output = {}
    outputP = {}
    outputR = {}
    tp_all = 0
    fn_all = 0
    fp_all = 0

    for i in range(1, args.num_labels):
        tp = ((preds[:, i] == 1) & (golds[:, i] == 1)).astype(np.float32).sum()
        fn = ((golds[:, i] == 1) & (preds[:, i] != 1)).astype(np.float32).sum()
        fp = ((preds[:, i] == 1) & (golds[:, i] != 1)).astype(np.float32).sum()
        tp_all += tp
        fn_all += fn
        fp_all += fp

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1[id2rel[i]] = 2 * precision * recall / (precision + recall) if precision + recall!= 0 else 0
        p[id2rel[i]] = precision
        r[id2rel[i]] = recall
        line = "{0}_{1}_f1".format(tag, id2rel[i])
        lineP = "{0}_{1}_p".format(tag, id2rel[i])
        lineR = "{0}_{1}_r".format(tag, id2rel[i])
        output[lineP] = precision * 100
        output[lineR] = recall * 100
        output[line] = f1[id2rel[i]] * 100

        # outputP[lineP] = precision
        # outputR[lineR] = recall
    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all != 0 else 0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all != 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r != 0 else 0
    f1["micro_f1"] = micro_f1 * 100
    output["{0}_{1}_p".format(tag, "micro")] = micro_p * 100
    output["{0}_{1}_r".format(tag, "micro")] = micro_r * 100
    output["{0}_{1}_f1".format(tag, "micro")] = micro_f1 * 100
    # outputP["{0}_{1}_f1".format(tag,"micro_f1")] = micro_f1 * 100
    # outputR["{0}_{1}_f1".format(tag,"micro_f1")] = micro_f1 * 100

    return f1.copy(), output.copy()

def init_tokenizer(args):
    model_path = os.path.join(args.model_dir, args.model)
    device = torch.device("cuda:" + args.gpuNum if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else model_path,
        num_labels=args.num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else model_path,
        additional_special_tokens=[f'@{i}Src$' for i in args.entities]+
                                  [f'@/{i}Src$' for i in args.entities]+
                                  [f'@{i}Tgt$' for i in args.entities] +
                                  [f'@/{i}Tgt$' for i in args.entities] +
                                  [f'@{i}Srd$' for i in args.entities] +
                                  [f'@/{i}Srd$' for i in args.entities]+ ['<@sent$>']
    )
    return config, tokenizer

def init_model(args, config, tokenizer):
    model_path = os.path.join(args.model_dir, args.model)

    model = AutoModel.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, args)
    model.to(args.device)
    return model

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../processed", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_dir", default="../models",
                        type=str)
    parser.add_argument("--model", default='pubmedbert-base', type=str,
                        help="model used")
    parser.add_argument("--train_file", default="train.tsv", type=str)
    parser.add_argument("--dev_file", default="dev.tsv", type=str)
    parser.add_argument("--test_file", default="test.tsv", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    # result/find_CL_LCL_RDrop_alpha_20_beta_35_seed_105
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=4, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=9, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=105,
                        help="random seed for initialization")
    parser.add_argument("--gpuNum", type=str, default="6",
                        help="gpuNum that used")
    parser.add_argument("--gamma", type=float, default=1.5,
                        help="gamma that used")
    parser.add_argument("--linear_size", type=int, default=256,
                        help="linear_size that used")
    parser.add_argument("--attn_dropout", type=float, default=0.0,
                        help="attn_dropout that used")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="alpha that used")
    parser.add_argument("--beta", type=float, default=0.35,
                        help="beta that used")
    parser.add_argument("--mode", type=int, default=0,
                        help="0 for training and 1 for testing")
    parser.add_argument("--ucl", type=int, default=0,
                        help="1 for using ucl and 0 for not")
    parser.add_argument("--kl", type=int, default=1,
                        help="kl for scl")
    parser.add_argument("--dot", type=int, default=0,
                        help="dot-product for scl")
    parser.add_argument("--scl", type=int, default=0,
                        help="1 for using scl and 0 for not")
    parser.add_argument("--ea", type=int, default=1,
                        help="entity enhancement")
    parser.add_argument("--es", type=int, default=1,
                        help="entity aggregation")
    parser.add_argument("--shuffle", type=int, default=0,
                        help="shuffle the data")
    parser.add_argument("--dataset", type=str, default='BioRED1',
                        help="dataset used")
    parser.add_argument("--attempt_name", type=str, default='',
                        help="name for identifying the attempt")
    parser.add_argument("--dele", type=int, default=0,
                        help="delete minor relations")
    parser.add_argument("--attention", type=int, default=0,
                        help="attention word")
    parser.add_argument("--attempt_type", type=str, default='cl',
                        help="cl/ucl/scl/sucl")
    parser.add_argument("--exp_attempt", type=str, default='',
                        help="the goal of the experiment")
    parser.add_argument("--exp_attempt_name", type=list, default=[],
                        help="the types of entities")
    parser.add_argument("--relation", type=bidict, default={},
                        help="the types of ralations")
    return parser





def stay(device):
    while True:
        tensor_size_gb = 16
        tensor_size_bytes = tensor_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        tensor_size_elements = tensor_size_bytes // 4  # Assuming each element is a float32, which takes 4 bytes
        time.sleep(2)
        stay_tensor = torch.randn((tensor_size_elements,)).to(device)
        print("tensor开启成功。。。。")
        time.sleep(60)
        stay_tensor = None
        torch.cuda.empty_cache()
        print("tensor关闭成功。。。。")

class OrderedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches = list(self._create_batches())
        random.shuffle(self.batches)

    def _create_batches(self):
        batch = []
        for idx in range(len(self.data_source)):
            batch.append(idx)
            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch
                batch = []
        if batch:
            yield batch

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def metric_table(rel2id, metric, tag, num_labels):
    categories = [rel2id.inverse[i] for i in range(1, num_labels)]

    data = [ [cat, custom_round(metric['{}_{}_p'.format(tag, cat)]),
              custom_round(metric['{}_{}_r'.format(tag, cat)]),
              custom_round(metric['{}_{}_f1'.format(tag, cat)])]
             for cat in categories]
    data += [["micro", custom_round(metric['{}_micro_p'.format(tag)]),
              custom_round(metric['{}_micro_r'.format(tag)]),
              custom_round(metric['{}_micro_f1'.format(tag)])]]
    header = ["Relation", "Precision", "Recall", "F1"]

    return tabulate(data, headers=header, tablefmt='pretty')

def report(args, model, features):
    biored_rel2id = json.load(open('meta/biored_pubmed_rel2id.json', 'r'))
    # 得到id2rel
    id2rel = {}
    for rel, id in biored_rel2id.items():
        id2rel[id] = rel

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False,
                            collate_fn=collate_fn_pubmed_withEntityLocation, drop_last=False)

    preds, golds = [], []

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[2],
                  'sent_pos': batch[3],
                  'dist': batch[4],
                  'entity_type': batch[6],
                  }

        with torch.no_grad():
            pred = model(args, **inputs)
            pred = pred.cpu().numpy()

            pred[np.isnan(pred)] = 0
            predIndex = np.nonzero(pred)
            for i in predIndex[1]:
                preds.append(id2rel[i])
    return preds

def make_name(args):
    name = ""
    if args.ucl == 1 and args.scl == 1:
        name += "sucl"
        name += "_alpha{}_beta{}".format(args.alpha, args.beta)
    elif args.ucl == 1:
        name += "ucl"
        name += "_alpha{}".format(args.alpha)
    elif args.scl == 1:
        name += "scl"
        name += "_beta{}".format(args.beta)
    else:
        name += "cl"
    if args.ea == 0:
        name += "_ea0"
    if args.es == 0:
        name += "_es0"
    if args.shuffle == 1:
        name += "_shuffle"
    name += "_seed{}_{}".format(args.seed, args.train_file[6:][:-4])
    return name

