from training import report, init_parser, set_seed, make_name, init_model, evaluate, metric_table
import os
import torch
import tqdm
from output import output, format_test, check_times

rel2id = {
    "NA": 0,
    "Association": 1,
    "Positive_Correlation": 2,
    "Bind": 3,
    "Negative_Correlation": 4,
    "Comparison": 5,
    "Cotreatment": 6
}

id2rel = {v: k for k, v in rel2id.items()}


start_tag = ['@GeneOrGeneProductSrc$', '@DiseaseOrPhenotypicFeatureSrc$', '@ChemicalEntitySrc$',
             '@GeneOrGeneProductTgt$', '@DiseaseOrPhenotypicFeatureTgt$', '@ChemicalEntityTgt$',
             '@OrganismTaxonSrc$', '@CellLineSrc$', '@SequenceVariantSrc$',
             '@OrganismTaxonTgt$', '@CellLineTgt$', '@SequenceVariantTgt$',
             '@GeneOrGeneProductSrd$', '@DiseaseOrPhenotypicFeatureSrd$', '@ChemicalEntitySrd$',
             '@OrganismTaxonSrd$', '@CellLineSrd$', '@SequenceVariantSrd$']
end_tag = ['@/GeneOrGeneProductSrc$', '@/DiseaseOrPhenotypicFeatureSrc$', '@/ChemicalEntitySrc$',
           '@/GeneOrGeneProductTgt$', '@/DiseaseOrPhenotypicFeatureTgt$', '@/ChemicalEntityTgt$',
           '@/OrganismTaxonSrc$', '@/CellLineSrc$', '@/SequenceVariantSrc$',
           '@/OrganismTaxonTgt$', '@/CellLineTgt$', '@/SequenceVariantTgt$',
           '@/GeneOrGeneProductSrd$', '@/DiseaseOrPhenotypicFeatureSrd$', '@/ChemicalEntitySrd$',
           '@/OrganismTaxonSrd$', '@/CellLineSrd$', '@/SequenceVariantSrd$']
start, end = {}, {}





# record the positions of mentions and sentence separator
def getEntityPos(tokens):
    src = [[], []]  # [[start],[end]]
    tgt = [[], []]  # [[start],[end]]
    sentenceSplitLabel_pos = []
    # print(tokens)
    # [CLS] token will be added in the beginning, so pos+1
    sents = []
    sent = []
    flag = 0
    flag1 = 0
    for i, token in enumerate(tokens):
        sent.append(token)
        if start.get(token):
            if token[-4:] == "Src$":
                if flag == 0:
                    flag = 1
                elif flag ==2:
                    flag = 3
                    flag1 = 1
            elif token[-4:] == "Tgt$":
                if flag == 0:
                    flag = 2
                elif flag == 1:
                    flag = 3
                    flag1 = 1
            else:
                flag = 3
                flag1 = 1
        elif token == '<@sent$>':
            sent.append(flag)
            sents.append(sent)
            flag = 0
            sent = []
    r = []
    for i in sents:
        if flag1==1:
            if i[-1] != 0:
                r += i[:-1]
        else:
            r += i[:-1]
    for i, token in enumerate(r):
        if start.get(token):
            if token[-4:] == "Src$":
                src[0].append(i + 1)
            elif token[-4:] == "Tgt$":
                tgt[0].append(i + 1)
            else:
                src[0].append(i + 1)
                tgt[0].append(i + 1)
        elif end.get(token):
            if token[-4:] == "Src$":
                src[1].append(i + 1)
            elif token[-4:] == "Tgt$":
                tgt[1].append(i + 1)
            else:
                src[1].append(i + 1)
                tgt[1].append(i + 1)
        elif token == '<@sent$>':
            sentenceSplitLabel_pos.append(i + 1)



    assert len(src[0]) == len(src[1]), "实体位置匹配错误"
    assert len(tgt[0]) == len(tgt[1]), "实体位置匹配错误"
    if len(src[0]) == 0:
        src = tgt.copy()
    if len(tgt[0]) == 0:
        tgt = src.copy()
    # assert len(src[0]) > 0 and len(tgt[0])>0, (src,tgt)
    return [src, tgt], sentenceSplitLabel_pos, r

# 读取pubtator格式，并且保存实体提及位置
def read(file_in, tokenizer, max_seq_length=1024):
    '''
    return  feature = {'input_ids': ids of tokenized input document with special tokens,
                       'entity_pos': list of positions of mentions  (beginning and end) ,
                       'sent_pos': list of positions of sentence separator ,
                       'dist': relatively minimal distance of source and target entities ,
                       'labels': multi-hot representation for relation label,
                       'entity_type': types of source and target entities,
                       }
    '''

    for i in start_tag:
        start[i] = True
    for i in end_tag:
        end[i] = True
    features = []
    labels = []
    with open(file_in, 'r', encoding='utf-8') as file_in:
        for line in tqdm.tqdm(file_in):
            _line = line.strip('\n')
            secs = _line.split("\t")
            inputText = secs[7]
            entity_type = [secs[1], secs[2]]
            dist = int(secs[6])
            dist = 19 if dist > 19 else dist
            # 处理label
            label = int(secs[8]) if secs[8].isdigit() else rel2id[secs[8]]
            # 处理inputIds
            inputTokens = tokenizer.tokenize(inputText)
            entity_pos, sentenceSplitLabel_pos, inputTokens = getEntityPos(inputTokens)  # [src,tgt],src=[[start],[end]]
            input_ids = tokenizer.convert_tokens_to_ids(inputTokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            newLabel = [0] * len(rel2id)
            newLabel[label] = 1
            # if len(input_ids)>512:
            #     print("pmid = "+secs[pmidIndex])
            #     print(len(inputTokens))
            feature = {'input_ids': input_ids,
                       'entity_pos': entity_pos,
                       'sent_pos': sentenceSplitLabel_pos,
                       'dist': dist,
                       'labels': newLabel,
                       'entity_type': entity_type,
                       'inputText':inputText,
                       }

            features.append(feature)
            labels.append(id2rel[label])
    return features, labels

def main(_input, device_num, load_path):
    if not os.path.exists(_input):
        print("file not exists")
        return
    args = init_parser().parse_args()
    args.gpuNum = device_num
    print(args)
    args.load_path = load_path
    args.train_batch_size = 1
    device, config, tokenizer, model = init_model(args)
    test_features, labels = read(_input, tokenizer, max_seq_length=args.max_seq_length)

    model.load_state_dict(torch.load(args.load_path, map_location=device))

    args.attempt_name = 'for_pred'
    f1, output = evaluate(args, model, test_features)
    print(metric_table(output,'dev'))
    # tp, fp, fn = 0, 0, 0
    # for i, j in zip(preds, labels):
    #     if i == j and i!= 'None':
    #         tp += 1
    #     else:
    #         if i == 'None':
    #             fn += 1
    #         else:
    #             fp += 1
    # print('tp:', tp, 'fp:', fp, 'fn:', fn)
    # print('precision:', tp/(tp+fp), 'recall:', tp/(tp+fn))
    # print('f1:', 2*tp/(2*tp+fp+fn))
    # print(sum([1 for i,j in zip(preds, labels) if i == j])/len(labels))
    # with open('result/pred.txt', 'w') as f:
    #     for i, j  in zip(preds, labels):
    #         f.write(j+'\t')
    #         f.write(i+'\n')

if __name__ == '__main__':
    _load_path = 'result/cl_seed77_train_scibert_mx_del_1_epoch14_64.25369252598743'
    main('../error/BioRED/pubmedbert-base/cl/cl_seed77_train_scibert_mx_del_1/test_scibert_cm_del.tsv', '4', _load_path)
    # while True:
    #     try:
    #         main(input("please enter a test instance:\n"), '1', _load_path)
    #     except Exception as e:
    #         continue


