from bidict import bidict

import torch
import random
import numpy as np
import json


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
def set_seed1(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if  torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn_pubmed_withEntityLocation(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    # print(max_len)
    # add [PAD] token to the end to make input consistent
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    sent_pos = [f["sent_pos"] for f in batch]

    # identify actual input

    input_ids = torch.tensor(input_ids, dtype=torch.long)

    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = [f["labels"] for f in batch]
    src_types = [f["src_types"] for f in batch]
    tgt_types = [f["tgt_types"] for f in batch]
    dist = [f["dist"] for f in batch]

    output = (input_ids, input_mask, entity_pos, sent_pos, labels, src_types, tgt_types, dist, [f['inputText'] for f in batch],[f['len'] for f in batch])
    return output


def collate_fn_pubmed(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    # labels = torch.tensor(labels, dtype=torch.float)
    output = (input_ids, input_mask, labels)
    return output


'''
加入实体类型
'''


def collate_fn2(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    entity_type = [f["entity_type"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts, entity_type, [f['inputText'] for f in batch])
    return output


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output


def replace2symbol(string):
    string = string.replace('”', '"').replace('’', "'").replace('–', '-').replace('‘', "'").replace('‑', '-').replace(
        '\x92', "'").replace('»', '"').replace('—', '-').replace('\uf8fe', ' ').replace('«', '"').replace(
        '\uf8ff', ' ').replace('£', '#').replace('\u2028', ' ').replace('\u2029', ' ')

    return string


def replace2space(string):
    spaces = ["\r", '\xa0', '\xe2\x80\x85', '\xc2\xa0', '\u2009', '\u2002', '\u200a', '\u2005', '\u2003', '\u2006',
              'Ⅲ', '…', 'Ⅴ', "\u202f"]

    for i in spaces:
        string = string.replace(i, ' ')
    return string

def read_metadata(dataset):
    with open(f'meta/{dataset}.json', 'r') as f:
        data = json.load(f)
    return data['entity'], bidict(data['relation']), data['alpha'], data['beta']
