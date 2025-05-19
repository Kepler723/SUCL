# def amplify(file):
#     info = {}
#     labels = {}
#     with open(file, 'r') as f:
#         with open(file[:-4] + '_more1.tsv', 'w') as g:
#             a=f.readlines()
#             g.writelines(a)
#             for line in a:
#                 label = line.split('\t')[6]
#                 info[label] = info.get(label, 0) +1
#                 if label not in labels:
#                     labels[label]=[line]
#                 else:
#                     labels[label].append(line)
#             print(info)
#             for i in info:
#                 if i=='None':
#                     continue
#                 else:
#                     if info[i] >400:
#                         g.writelines(labels[i])
#                     else:
#                         times = 400//info[i]+1
#                         g.writelines(labels[i]*times)
#     with open(file[:-4] + '_more1.tsv', 'r') as f:
#         print(len(f.readlines()))
# amplify('processed/BioRED/train3.tsv')
def amplify(file):
    add_more = []
    count = 0
    with open(file, 'r') as f:
        with open(file[:-4] + '_more.tsv', 'w') as g:
            a=f.readlines()
            # print(a[1])
            g.writelines(a)
            for line in a:
                label = line.split('\t')[6]
                if label != 'None':
                    # print(label)
                    add_more.append(line)
                else:
                    count += 1
            print(count)
            print(len(add_more))
            # g.writelines(add_more)

    # with open(file[:-4] + '_more.tsv', 'r') as f:
    #     print(len(f.readlines()))
amplify('processed/Bc8/train/train_dev_more.tsv')