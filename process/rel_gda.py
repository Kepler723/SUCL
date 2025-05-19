import cupy
import os
from tqdm import tqdm
import spacy
import nltk
from nltk import sent_tokenize

source = 'origin'
processed = 'processed'


entities = ["Gene","Disease"]

class Passage:
    def __init__(self, pmid, text):
        self.pmid = pmid
        self.text = text
        self._text = ''
        self.id2type = {}
        self.type2id = {'Disease': [], 'Gene': []}
        self.id2loc = {}
        self.ids2re = {}

    def add_entity(self, type, start, offset, id):
        self.id2type[id] = type
        self.type2id[type].append(id)
        if self.id2loc.get(id):
            self.id2loc[id].append((start, offset))
        else:
            self.id2loc[id] = [(start, offset)]

    def add_relation(self, id1, id2, rel):
        self.ids2re[(id1, id2)] = rel

pairs = ["Gene/Disease"]


def load_data(dataset, data_type):
    file_path = os.path.join(source, dataset, data_type, 'abstracts.txt')
    with open(file_path, 'r') as f:
        docs=[]
        temp=''
        for line in f.readlines():
            if line.strip()=='':
                if temp!='':
                    docs.append(temp)
                    temp = ''
            else:
                temp += line
        if temp!='':
            docs.append(temp)
    rel_path = os.path.join(source, dataset, data_type,'labels.csv')
    with open(rel_path, 'r') as f:
        rels = f.read().split('\n')[1:]
        if rels[-1] == '':
            rels = rels[:-1]
    entity_path = os.path.join(source, dataset, data_type, 'anns.txt')
    with open(entity_path, 'r') as f:
        entitys = f.read().split('\n\n')
    sent_path = os.path.join(source, dataset, data_type, 'sentences.txt')
    with open(sent_path, 'r') as f:
        sents = f.read().split('\n\n')
    return docs, rels, entitys, sents

def add_label(pmid, text, src_type, des_type, src, des, sentence_len):

    src = [(i[0], i[1], '{}Src'.format(src_type)) for i in src]
    des = [(i[0], i[1], '{}Tgt'.format(des_type)) for i in des]

    combined = src + des + sentence_len
    combined = sorted(combined, key=lambda x: x[0])
    assert combined[-1][2]=='<@sent$>', (pmid, combined[-1])
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


def add_entity(entity, passage):

    for line in entity.split('\n'):
        line = line.split('\t')

        start, end, name, _type, identifier = line[1], line[2], line[3], line[4], line[5]

        start, end = int(start), int(end)
        for _id in set(identifier.split('|')):
            passage.add_entity(_type, start, end, _id)


def handle_doc(passage):
    sentence_len = split_sents(passage.text, sent_model)
    sentence_len = [(i,i,'<@sent$>') for i in sentence_len]

    records = set()
    out = []
    for pair in pairs:
        types = pair.split('/')
        src = types[0]
        des = types[1]
        if passage.type2id.get(src) is None or passage.type2id.get(des) is None:
            continue
        for j in passage.type2id[src]:
            if passage.id2type.get(j) is None:
                continue
            for k in passage.type2id[des]:
                if (j, k) in records or (k, j) in records or passage.id2type.get(k) is None:
                    continue
                records.add((j, k))
                mention_src = passage.id2loc[j]
                mention_des = passage.id2loc[k]

                text1, min_dis = add_label(passage.pmid, passage.text, src, des, mention_src, mention_des, sentence_len)
                reType = 'None'
                if passage.ids2re.get((j, k)):
                    reType = passage.ids2re[(j, k)]
                elif passage.ids2re.get((k, j)):
                    reType = passage.ids2re[(k, j)]
                out.append('\t'.join([passage.pmid, src, des, j, k, str(min_dis == 0), str(min_dis), text1.replace('  ', ' '), str(reType), "None"])+'\n')
    return out

def handle_docs(dataset, file_type, docs, rels, entitys, sents):
    out_path = os.path.join(processed, dataset, '{}.tsv'.format(file_type))
    plist ={}
    info = {}
    for doc in tqdm(docs):
        lines = doc.split('\n')
        pmid = lines[0].strip()
        text = lines[1]
        if len(lines) == 3:
            text += ' ' + lines[2]
        passage = Passage(pmid, text)
        plist[pmid] = passage
    for entity in entitys:
        pmid = entity.split('\n')[0].split('\t')[0]
        add_entity(entitys, plist[pmid])
    for rel in rels:
        rel = rel.split(',')
        pmid, id1, id2, rel_type   = rel[0], rel[1], rel[2], rel[3]
        plist[pmid].add_relation(id1, id2, rel_type)

    with open(out_path, 'w') as f:
        for pmid, passage in tqdm(plist.items()):
            temp = handle_doc(passage)
            f.writelines(temp)
    print(info)



# def clean_passage(doc):
if __name__ == '__main__':
    _dataset = 'GDA'
    _file_types = ['train','test']

    for _file_type in _file_types:
        _docs,_rels, _entitys, _sents = load_data(_dataset, _file_type)
        print(len(_docs), len(_rels), len(_entitys), len(_sents))
        # handle_docs(_dataset, _file_type, _docs, _rels, _entitys, _sents)
