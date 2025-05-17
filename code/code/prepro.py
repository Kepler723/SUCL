from tqdm import tqdm
import ujson as json
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

# record the positions of mentions and sentence separator
def get_mask(start, end, tokens):

    src, tgt = [], []
    sent_pos = []
    _tokens = []
    flag = 1
    deleted = 0
    for i, token in enumerate(tokens):
        if token in start:
            if token[-4:] == "Src$":
                src.append(i-deleted+1)
            elif token[-4:] == "Tgt$":
                tgt.append(i-deleted+1)
            else:
                src.append(i-deleted+1)
                tgt.append(i-deleted+1)
            _tokens.append(token)
            flag = 0
        elif token in end:
            deleted += 1
            flag = 1
            continue
        elif token == '<@sent$>':
            sent_pos.append(i-deleted+1)
        if flag == 1:
            _tokens.append(token)
        else:
            deleted += 1

    return [src, tgt], sent_pos, _tokens

def get_entity(start, tokens):

    src, tgt = [], []
    sent_pos = []

    for i, token in enumerate(tokens):
        if token in start:
            if token[-4:] == "Src$":
                src.append(i + 1)
            elif token[-4:] == "Tgt$":
                tgt.append(i + 1)
            elif token[-4:] == "Srd$":
                src.append(i + 1)
                tgt.append(i + 1)

        elif token == '<@sent$>':
            sent_pos.append(i + 1)
            
    return [src, tgt], sent_pos

# 读取pubtator格式，并且保存实体提及位置
def read_biored_pubmed(file_in, tokenizer, args):
    start = set([f'@{i}Src$' for i in args.entities] + [f'@{i}Tgt$' for i in args.entities] +  [f'@{i}Srd$' for i in args.entities])
    features = []
    if file_in == "":
        return None
    with open(file_in, 'r') as infile:
        for line in tqdm(infile):
            if line.strip() == "":
                continue
            _line = line.strip("\n")
            secs = _line.split("\t")
            dist = int(secs[5])
            src_type, tgt_type = secs[1], secs[2]
            newLabel = [0] * args.num_labels
            newLabel[args.relations[secs[6]]] = 1

            inputText = secs[7]
            inputTokens = tokenizer.tokenize(inputText)

            entity_pos, sent_pos = get_entity(start, inputTokens)  # [src,tgt],src=[[start],[end]]
            if entity_pos[0]==[] or entity_pos[1]==[]:
                print(secs[0],inputTokens)
            input_ids = tokenizer.convert_tokens_to_ids(inputTokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)


            feature = {'input_ids': input_ids,
                       'entity_pos': entity_pos,
                       'sent_pos': sent_pos,
                       'labels': newLabel,
                       'src_types': src_type,
                       'tgt_types': tgt_type,
                       'dist': dist,
                       'inputText':line,
                       'len':len(input_ids),
                       }
            features.append(feature)

    return features