import ujson as json
from tqdm import tqdm
import  os

rel_file = 'processed_result.txt'
times_json = '/home/gzwang/docre-cl/times.json'

def format_test(test_json, rels, test_result):
    with open(test_json, 'r') as file:
        data = json.load(file)
        for document in tqdm(data['documents'], total=len(data['documents']), desc='writing results into file'):
            relations = rels.get(document['id'], [])
            for i, rel in enumerate(relations):
                document['relations'].append(
                    {
                        "id": "R{}".format(i),
                        "infons": {
                            "entity1": rel[0],
                            "entity2": rel[1],
                            "type": rel[2],
                            "novel": "Novel"
                        },
                        "nodes": []
                    },
                )
        with open('format/{}.json'.format(test_result), 'a') as f:
            json.dump(data, f)


def output(test_tsv, label_file):
    dic = {}
    pairs = {}
    f1 = open(test_tsv, 'r')
    f2 = open(label_file, 'r')
    file1 = f1.readlines()
    file2 = f2.readlines()
    l1, l2 = len(file1), len(file2)
    assert l1 == l2, '预测结果长度不正确 {}/{}'.format(l1, l2)
    with open('processed_result.txt', 'w') as processed_result:
        for line1, line2 in zip(file1, file2):
            line2 = line2[:-1]
            if line2 == 'None':
                continue
            line1 = line1[:-1].split('\t')
            processed_result.write('{}\t{}\t{}\t{}\n'.format(line1[0], line1[3], line1[4], line2))
            if pairs.get('{}/{}'.format(line1[1],line1[2])):
                pairs['{}/{}'.format(line1[1],line1[2])]+=1
            else:
                pairs['{}/{}'.format(line1[1],line1[2])]=1
            if dic.get(line1[0]):
                dic[line1[0]].append([line1[3], line1[4], line2])
            else:
                dic[line1[0]] = [[line1[3], line1[4], line2]]
        f1.close()
        f2.close()
    return dic


def check_times(item):
    with open(times_json, 'r') as f:
        data = json.load(f)
        temp = data.get(item, 0)
        data[item] = temp + 1
    with open(times_json, 'w') as f:
        json.dump(data, f)
    return temp

