import cupy
import os

from idna.idnadata import joining_types
from tqdm import tqdm
import spacy
import nltk
from nltk import sent_tokenize

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

entities = ["Chemical","Disease"]
alias = {
    'Disease': 'DiseaseOrPhenotypicFeature',
    'Organism': 'OrganismTaxon'
}
class Passage:
    def __init__(self, pmid, text):
        self.pmid = pmid
        self.text = text
        self._text = ''
        self.id2type = {}
        self.type2id = {'Chemical': [], 'Disease': [], 'Gene': []}
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

pairs = ["Chemical/Disease",
         "Chemical/Gene", "Disease/Gene"]

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

def load_data(dataset, data_type):
    file_path = os.path.join(source, dataset, '{}_abstracts.txt'.format(data_type))
    with open(file_path, 'r') as f:
        docs = f.read().split('\n\n')
        if docs[-1] == '':
            docs=docs[:-1]
    rel_path = os.path.join(source, dataset, '{}_relationships.tsv'.format(data_type))
    with open(rel_path, 'r') as f:
        rels = f.read().split('\n')
        if rels[-1] == '':
            rels = rels[:-1]
        return docs, rels

def is_spacy_model(model):
    return isinstance(model, spacy.language.Language)

def split_sents( text, sent_model):
    doc = sent_model(text)
    # for i in doc.sents:
    #     print(i.sent)
    sent_len = [-1]
    sen_len = []
    for i, sent in enumerate(doc.sents):
        sent_len.append(sent_len[-1] + len(sent.text))
        # end_note.add(sent.text[-1])
        l = len(sent.text)

        while text[sent_len[-1] - l + 1:sent_len[-1] + 1] != sent.text:
            sent_len[-1] += 1
    sent_len = sent_len[1:]
    for i in range(len(sent_len)):
        if text[sent_len[i]] not in ['.', '?', '!', '"', "'"] and i != len(sent_len) - 1:
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
    assert combined[-1][2] == '<@sent$>', (pmid, combined[-1])
    # print(combined)
    combined1 = []
    i = 0
    while i < len(combined) - 1:
        if combined[i][2] != '<@sent$>' and combined[i + 1][2] == '<@sent$>':
            combined1.append(combined[i])
            if combined[i + 1][1] < combined[i][1]:
                i += 1
        elif combined[i][0] == combined[i + 1][0]:
            if combined[i][1] == combined[i + 1][1]:
                assert combined[i][2] != '<@sent$>' and combined[i + 1][2] != '<@sent$>', (
                pmid, combined[i], combined[i + 1])
                combined1.append((combined[i][0], combined[i][1], '{}Srd'.format(src_type)))
                i += 1
            else:
                raise ValueError(pmid, combined[i], combined[i + 1])
        else:
            if combined[i][2] == '<@sent$>':
                assert combined[i][0] < combined[i + 1][0], (pmid, combined[i], combined[i + 1])
                combined1.append(combined[i])
            else:
                assert combined[i][1] <= combined[i + 1][0], (pmid, combined[i], combined[i + 1])
                combined1.append(combined[i])
        i += 1
    combined1.append(combined[-1])
    # print(combined1)
    min_dis = len(sentence_len)
    result = []
    temp = []
    for tag in combined1[::-1]:
        if tag[2] == '<@sent$>':
            text = text[:tag[0] + 1] + ' <@sent$> ' + text[tag[0] + 1:]
            result.append(temp)
            temp = []
        else:
            text = text[:tag[0]] + ' @{}$ '.format(tag[2]) + text[tag[0]:tag[1]] + ' @/{}$ '.format(tag[2]) + text[
                                                                                                              tag[1]:]
            temp.append(tag[2])
    src_idx, des_idx = len(sentence_len) - 1, len(sentence_len) - 1
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
    return text.replace('  ', ' '), min_dis


def add_entity(doc):
    lines = doc.split('\n')
    temp = lines[0].split('|')
    pmid, title = temp[0], temp[2]
    if len(lines[1].split('|'))!=3:
        text = title
        begin = 1
    else:
        text = title + ' ' + lines[1].split('|')[2]
        begin = 2

    passage = Passage(pmid, text)

    for line in lines[begin:]:
        line = line.split('\t')
        if len(line) >= 6:
            start, end, name, _type, identifier = line[1], line[2], line[3], line[4], line[5]

            start, end = int(start), int(end)
            for _id in set(identifier.split('|')):
                passage.add_entity(_type, start, end, _id)
        # else:
        #     _, _type, id1, id2 = line
        #
        #     if '{}/{}'.format(id2type[id2], id2type[id1]) in _pairs and id2type[id2]!=id2type[id1]:
        #         ids2re[(id2, id1)] = rel2ids[_type]
        #     else:
        #         ids2re[(id1, id2)] = rel2ids[_type]
    return pmid, passage

def handle_doc(passage, sent_model):
    try:
        sentence_len = split_sents(passage.text, sent_model)
        sentence_len = [(i,i,'<@sent$>') for i in sentence_len]
    except:
        sentence_len = split_sents(passage.text, lg_model)
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
                out.append('\t'.join([passage.pmid, src, des, j, k, str(min_dis), reType, text1.replace('  ', ' ')])+'\n')
    return out

def handle_docs(dataset, file_type, docs, rels, model):
    out_path = os.path.join(processed, dataset, '{}.tsv'.format(file_type))
    plist ={}
    info = {}
    for index, doc in enumerate(tqdm(docs)):
        pmid, passage = add_entity(doc)
        plist[pmid] = passage
    for rel in rels:
        rel = rel.split('\t')
        pmid, rel_type, id1, id2  = rel[0], rel[1], rel[2], rel[3]
        pair, rel_type = rel_type.split(':')
        plist[pmid].add_relation(id1, id2, rel_type)
        info[pair] = info.get(pair, 0) + 1

    with open(out_path, 'w') as f:
        for pmid, passage in tqdm(plist.items()):
            temp = handle_doc(passage, model)
            f.writelines(temp)
    print(info)



# def clean_passage(doc):
if __name__ == '__main__':
    _datasets = ['chem_dis_gene']
    _file_types = ['train','dev','test']
    spacy.require_gpu(2)
    _model = spacy.load('en_core_sci_scibert')
    lg_model = spacy.load('en_core_sci_lg')

    for _dataset in _datasets:
        for _file_type in _file_types:
            _docs,_rels = load_data(_dataset, _file_type)
            handle_docs(_dataset, _file_type, _docs, _rels, _model)
    # with open('processed/BioRED1/Dev_lg_cm.tsv','r') as f:
    #     count = 0
    #     for line in f.readlines():
    #         if line.strip()=='':
    #             count += 1
    #     print(count)
