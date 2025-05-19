import cupy
import os

from idna.idnadata import joining_types
from tqdm import tqdm
import spacy
import nltk
from nltk import sent_tokenize
import json
import copy
source = 'origin'
processed = 'processed'
pair2rel = {
    "ChemicalEntity/ChemicalEntity": [3],
    "ChemicalEntity/DiseaseOrPhenotypicFeature": [3, 5, 6, 7, 8],
    "ChemicalEntity/GeneOrGeneProduct": [5, 6, 7, 8],
    "DiseaseOrPhenotypicFeature/GeneOrGeneProduct": [3, 5, 6, 7, 8],
    "GeneOrGeneProduct/GeneOrGeneProduct": [5, 6, 7, 8],
    "ChemicalEntity/SequenceVariant": [3, 5, 6, 7, 8],
    "DiseaseOrPhenotypicFeature/SequenceVariant": [3, 5, 6, 7, 8]
}
# entities = ["ChemicalEntity", "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature",
#              'OrganismTaxon', "CellLine", "SequenceVariant"]

entities = ['LOC', 'ORG', 'MISC', 'PER', 'TIME', 'NUM']


# pairs = ["ChemicalEntity/ChemicalEntity", "ChemicalEntity/DiseaseOrPhenotypicFeature",
#          "ChemicalEntity/GeneOrGeneProduct", "DiseaseOrPhenotypicFeature/GeneOrGeneProduct",
#          "GeneOrGeneProduct/GeneOrGeneProduct", "ChemicalEntity/SequenceVariant",
#          "DiseaseOrPhenotypicFeature/SequenceVariant"]
pairs = ['LOC/LOC', 'ORG/LOC', 'ORG/ORG', 'MISC/LOC', 'PER/LOC', 'ORG/TIME', 'PER/ORG', 'MISC/PER', 'MISC/ORG', 'MISC/TIME', 'PER/TIME', 'MISC/MISC', 'LOC/ORG', 'PER/MISC', 'LOC/TIME', 'LOC/MISC', 'LOC/PER', 'PER/PER', 'ORG/PER', 'ORG/MISC', 'TIME/TIME', 'TIME/PER', 'NUM/PER', 'TIME/LOC', 'LOC/NUM', 'NUM/LOC']

count =0
pair = {}
rel2ids = {
    "None": 0,
    "CID": 1
    }
# rel2ids = {
#     "None": 0,
#     "Association": 1,
#     "Positive_Correlation": 2,
#     "Bind": 3,
#     "Negative_Correlation": 4,
#     "Comparison": 5,
#     "Conversion": 6,
#     "Cotreatment": 7,
#     "Drug_Interaction": 8
# }

# mix_pairs = ["ChemicalEntity/ChemicalEntity", "ChemicalEntity/DiseaseOrPhenotypicFeature",
#              "ChemicalEntity/GeneOrGeneProduct", "DiseaseOrPhenotypicFeature/GeneOrGeneProduct",
#              "GeneOrGeneProduct/GeneOrGeneProduct", "ChemicalEntity/SequenceVariant",
#              "DiseaseOrPhenotypicFeature/SequenceVariant", "SequenceVariant/SequenceVariant",
#              "GeneOrGeneProduct/SequenceVariant"]



def is_spacy_model(model):
    return isinstance(model, spacy.language.Language)

def split_sents( text, sent_model):
    global counts
    doc = sent_model(text)
    if is_spacy_model(sent_model):
        doc = [sent.text for sent in doc.sents]
    sent_len = [-1]
    sen_len = []
    for i, sent in enumerate(doc):
        sent_len.append(sent_len[-1] + len(sent))
        # end_note.add(sent.text[-1])
        l = len(sent)
        while text[sent_len[-1]-l+1:sent_len[-1]+1]!=sent:
            sent_len[-1] += 1
    sent_len = sent_len[1:]
    for i in range(len(sent_len)):
        if text[sent_len[i]] not in ['.', '?', '!', '"', "'"] and i != len(list(doc)) - 1:
            pass
        else:
            sen_len.append(sent_len[i])
    return sen_len

def add_label(pmid, text, src_type, des_type, src, des, sentence_len):
    src = [(i[0], i[1], '{}Src'.format(src_type)) for i in src]
    des = [(i[0], i[1], '{}Tgt'.format(des_type)) for i in des]
    # print('src',src)
    # print('des',des)
    combined = src + des + sentence_len
    combined = sorted(combined, key=lambda x: x[0])
    assert combined[-1][2]=='<@sent$>', (pmid, combined[-1])
    # print(combined)
    combined1 = []
    i = 0
    while i <len(combined)-1:
        if combined[i][2]!='<@sent$>' and combined[i+1][2]=='<@sent$>':
            combined1.append(combined[i])
            if combined[i][0]<=combined[i+1][1]<=combined[i][1]:
                i+=1
        elif combined[i][0] == combined[i + 1][0]:
            if combined[i][1] == combined[i + 1][1]:
                assert combined[i][2]!='<@sent$>' and combined[i+1][2]!='<@sent$>', (pmid, combined[i], combined[i + 1])
                combined1.append((combined[i][0], combined[i][1], '{}Srd'.format(src_type)))
                i+=1
            else:
                raise ValueError(pmid, combined[i], combined[i + 1])
        else:
            if combined[i][2] == '<@sent$>':
                assert combined[i][0] < combined[i + 1][0], (pmid, combined[i], combined[i + 1])
                combined1.append(combined[i])
            else:
                assert combined[i][1] <= combined[i + 1][0], (pmid, combined[i], combined[i + 1])
                combined1.append(combined[i])
        i+=1
    combined1.append(combined[-1])
    # print(combined1)
    min_dis = len(sentence_len)
    result = []
    temp = []
    for tag in combined1[::-1]:
        if tag[2] == '<@sent$>':
            text = text[:tag[0]+1] + ' <@sent$> ' + text[tag[0]+1:]
            result.append(temp)
            temp = []
        else:
            text = text[:tag[0]] + ' @{}$ '.format(tag[2]) + text[tag[0]:tag[1]] + ' @/{}$ '.format(tag[2]) + text[tag[1]:]
            temp.append(tag[2])
    src_idx, des_idx =len(sentence_len)-1, len(sentence_len)-1
    for i, item in enumerate(result):
        for j in item:
            if j == '{}Src'.format(src_type):
                src_idx = i
                min_dis = min(min_dis, abs(src_idx - des_idx))
            elif j == '{}Tgt'.format(des_type):
                des_idx = i
                min_dis = min(min_dis, abs(src_idx - des_idx))
            if min_dis == 0:
                break
        if min_dis == 0:
            break
    return text.replace('  ',' '), min_dis


def handle_doc(pmid,doc):

    id2loc = {}
    ids2re = {}
    type2id = {_:set() for _ in entities}
    id2type = {}
    id_name = {}
    for index,entity in enumerate(doc['vertexSet']):
        id2loc[index] = []
        for mention in entity:
            start, end, sent_id = mention['pos'][0], mention['pos'][1], mention['sent_id']
            id2loc[index].append((start, end, sent_id))
            _type = mention['type']
            if _type not in type2id:
                type2id[_type] = set()
            id2type[index] = _type
            type2id[_type].add(index)
    for rel in doc.get('labels', []):
        ids2re[(rel['h'], rel['t'])] = rel['r']
    sents = [ [word for word in sent] for sent in doc['sents']]


    # sentence_len = split_sents(text, sent_model)
    # sentence_len = [(i,i,'<@sent$>') for i in sentence_len]
    records = set()
    out = []
    for pair in pairs:
        types = pair.split('/')
        src = types[0]
        des = types[1]
        # print(src, des)
        for j in type2id[src]:
            if id2type.get(j) is None:
                continue
            for k in type2id[des]:
                if (j, k) in records or (k, j) in records or id2type.get(k) is None:
                    continue
                records.add((j, k))
                # print(j, k)
                mention_src = copy.deepcopy(id2loc[j])
                mention_src = [(i[0], i[1], i[2] ,'{}Src'.format(src)) for i in mention_src]
                mention_des = copy.deepcopy(id2loc[k])
                mention_des = [(i[0], i[1], i[2], '{}Tgt'.format(des)) for i in mention_des]
                combined = mention_src + mention_des
                combined.sort(key=lambda x: (x[2],x[1]), reverse=True)
                sents1 = copy.deepcopy(sents)
                # print(sents1)
                min_dis = len(sents1)
                for i in combined:
                    # print(i)
                    sents1[i[2]].insert(i[1], '@/{}$'.format(i[3]))
                    sents1[i[2]].insert(i[0], '@{}$'.format(i[3]))
                src_idx, des_idx = len(sents1) - 1, len(sents1) - 1
                for i, item in enumerate(combined):
                        if item[3] == '{}Src'.format(src):
                            src_idx = i
                        else:
                            des_idx = i
                        min_dis = min(min_dis, abs(src_idx - des_idx))
                        if min_dis == 0:
                            break
                reType = 'None'
                if ids2re.get((j, k)):
                    reType = ids2re[(j, k)]
                elif ids2re.get((k, j)):
                    reType = ids2re[(k, j)]
                text = ' <@sent$> '.join([' '.join(sent) for sent in sents1]) +' <@sent$>'
                out.append('\t'.join([str(pmid), src, des, str(j), str(k), str(min_dis == 0), str(min_dis), text.replace('  ', ' '), reType, "None"])+'\n')
    return out

def handle_docs(dataset, file_type):
    in_path = os.path.join(source, dataset, '{}.json'.format(file_type))
    out_path = os.path.join(processed, dataset, '{}.tsv'.format(file_type))
    with open(in_path, 'r') as f:
        with open(out_path, 'w') as out:
            docs = json.load(f)
            for i, doc in tqdm(enumerate(docs),total=len(docs),desc='Processing'):
                out.writelines(handle_doc(i,doc))



# def clean_passage(doc):
if __name__ == '__main__':
    _datasets = ['DocRed']
    _file_types = ['train_distant']
    spacy.require_gpu(0)
    model = spacy.load('en_core_sci_scibert')
    for _dataset in _datasets:
        for _file_type in _file_types:
            handle_docs(_dataset, _file_type)
    # print(pair)
    temp = []
    for i in pair.keys():
        temp.append('{}/{}'.format(i[0], i[1]))
    # print(temp)
    # print(entities)

    # with open('processed/BioRED1/Dev_lg_cm.tsv','r') as f:
    #     count = 0
    #     for line in f.readlines():
    #         if line.strip()=='':
    #             count += 1
    #     print(count)
