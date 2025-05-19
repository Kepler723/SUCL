import cupy
import os

from idna.idnadata import joining_types
from tqdm import tqdm
import spacy
import nltk
from nltk import sent_tokenize

source = 'origin'
processed = 'processed'

entities = ["Chemical","Disease"]
pairs = ["Chemical/Disease"]

rel2ids = {
    "None": 0,
    "CID": 1
    }
c={}
def read_bio(path):

    with open(path, 'r') as f:
        for line in f:
            line = line.split('\t')
            pmid = line[0]
            src,tgt = line[3], line[4]
            c[pmid] = c.get(pmid, [])+[(src, tgt)]
    return c
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
    file_path = os.path.join(source, dataset, '{}.PubTator'.format(data_type))
    with open(file_path, 'r') as f:
        data = f.read().split('\n\n')
        if data[-1] == '':
            data=data[:-1]
        return data

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

        while text[sent_len[-1]-l+1:sent_len[-1]+1]!=sent.text:
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
    assert combined[-1][2]=='<@sent$>', (pmid, combined[-1])
    # print(combined)
    combined1 = []
    i = 0
    while i <len(combined)-1:
        if combined[i][2]!='<@sent$>' and combined[i+1][2]=='<@sent$>':
            combined1.append(combined[i])
            if combined[i+1][1]<combined[i][1]:
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



def handle_doc(doc, sent_model):
    lines = doc.split('\n')
    temp = lines[0].split('|')
    pmid, title = temp[0], temp[2]
    if title == '- No Title -':
        return []
    abstract = lines[1].split('|')[2]
    text = title + ' ' + abstract
    # print(text)
    _pairs = pairs
    id2loc = {}
    ids2re = {}
    type2id = {i: set() for i in entities}
    id2type = {}
    for line in lines[2:]:
        line = line.split('\t')
        # print(line)
        if len(line) >= 6:
            start, end, name, _type, identifier = line[1], line[2], line[3], line[4], line[5]

            start, end = int(start), int(end)
            for _id in set(identifier.split('|')):
                if _id not in id2loc:
                    id2loc[_id] = [(start, end)]
                else:
                    id2loc[_id].append((start, end))
                # print(start, end, name, text[start:end], name == text[start:end])
                id2type[_id] = _type
                type2id[_type].add(_id)
        else:
            _, _type, id1, id2 = line

            if '{}/{}'.format(id2type[id2], id2type[id1]) in _pairs and id2type[id2]!=id2type[id1]:
                ids2re[(id2, id1)] = _type
            else:
                ids2re[(id1, id2)] = _type
    text = text.replace('..', '.')
    sentence_len = split_sents(text, sent_model)
    sentence_len = [(i,i,'<@sent$>') for i in sentence_len]
    records = set()
    out = []
    if c.get(pmid) is None:
        return []
    for pair in c[pmid]:
        id_src, id_tgt = pair
        src = id2type.get(id_src)
        des = id2type.get(id_tgt)
        mention_src = id2loc[id_src]
        mention_des = id2loc[id_tgt]
        text1, min_dis = add_label(pmid, text, src, des, mention_src, mention_des, sentence_len)


        reType = 'None'
        if ids2re.get((id_src, id_tgt)):
            reType = ids2re[(id_src, id_tgt)]
        elif ids2re.get((id_tgt,id_src)):
            reType = ids2re[(id_tgt, id_src)]
        out.append('\t'.join([pmid, src, des, id_src, id_tgt, str(min_dis), reType, text1.replace('  ', ' ')]) + '\n')
    return out

def handle_docs(dataset, file_type, docs, sent_model):
    out_path = os.path.join(processed, dataset, '{}_bioredirect.tsv'.format(file_type))
    with open(out_path, 'w') as out:
        for doc in tqdm(docs,total=len(docs),desc='Processing'):
            # handle_doc(doc, mode, sent_model)
            out.writelines(handle_doc(doc, sent_model))


# def clean_passage(doc):
if __name__ == '__main__':
    _datasets = ['CDR']
    _file_types = ['test']
    _models = 'scibert' # 'lg', 'scibert', 'nltk'
    spacy.require_gpu(5)
    read_bio('./bioredirect/datasets/cdr/processed/test.tsv')

    model = spacy.load('en_core_sci_scibert')
    for _dataset in _datasets:
        for _file_type in _file_types:
            _docs = load_data(_dataset, _file_type)
            handle_docs(_dataset, _file_type, _docs, model)
    # with open('processed/BioRED1/Dev_lg_cm.tsv','r') as f:
    #     count = 0
    #     for line in f.readlines():
    #         if line.strip()=='':
    #             count += 1
    #     print(count)
