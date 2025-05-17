import torch
import torch.nn.functional as F
import numpy as np


def cut_long_seq(input_ids, attention_mask, start_tokens, end_tokens, l_token):
    if l_token <= 1022:
        clips = [(0, 511), (l_token - 511, 511)]
    elif l_token <= 1532:
        clips = [(0, 511), (511 - (1532 - l_token) // 2, 510), (l_token - 511, 511)]
    elif l_token <=2042:
        clips = [(0, 511), (511 - (2042 - l_token) // 2, 510), (1021 - (2042 - l_token) // 2, 510),
                 (l_token - 511, 511)]
    else:
        clips = [(0, 511), (511 - (2552 - l_token) // 2, 510), (1021 - (2552 - l_token) // 2, 510),
                 (1531-(2552-l_token) // 2, 510), (l_token - 511, 511)]
    temp = []
    mask =[]
    for i in range(len(clips)):
        if i == 0:
            temp.append(torch.cat([input_ids[clips[i][0]:clips[i][0]+clips[i][1]], end_tokens], dim=-1))
            mask.append(torch.cat([attention_mask[clips[i][0]:clips[i][0]+clips[i][1]],torch.ones(1).to(attention_mask)],dim=-1))
        elif i == len(clips) - 1:
            temp.append(torch.cat([start_tokens, input_ids[clips[i][0]:clips[i][0]+clips[i][1]]], dim=-1))
            mask.append(torch.cat([torch.ones(1).to(attention_mask),attention_mask[clips[i][0]:clips[i][0]+clips[i][1]]],dim=-1))
        else:
            temp.append(torch.cat([start_tokens, input_ids[clips[i][0]:clips[i][0]+clips[i][1]], end_tokens], dim=-1))
            mask.append(torch.cat([torch.ones(1).to(attention_mask),attention_mask[clips[i][0]:clips[i][0]+clips[i][1]],torch.ones(1).to(attention_mask)],dim=-1))
    return clips, temp, mask, len(clips)


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    # 处理长文本的方法
    n, c = input_ids.size()
    # print('序列长度最大值：{}'.format(c))
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    cls_output = []
    # les than 512 tokens
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        ''' sequence_output[batch_size,sequence_len,hidden_size]:
                last_hidden_state,each token represented in hidden_size length
            attention[batch_size,num_heads,sequence_len,sequence_len]:
                last layer of attention
        '''
        sequence_output = output[0]
        return sequence_output
    else:
        new_clips, new_input_ids, new_attention_mask, num_seg = [], [], [], []
        # list of lengths of a batch of input sequences
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        # 将长文本分割成两个相交的短文本
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
                new_clips.append([(0,l_i)])
            else:

                _clips, _input_ids, _mask, _seg = cut_long_seq(input_ids[i], attention_mask[i], start_tokens, end_tokens, l_i)
                new_input_ids.extend(_input_ids)
                new_attention_mask.extend(_mask)
                num_seg.append(_seg)
                new_clips.append(_clips)

        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)

        # 对短文本进行编码
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        sequence_output = output[0]

        i = 0
        new_output = []

        for (n_s, l_i,l_seq) in zip(num_seg, new_clips,seq_len):
            merged_cls = []
            output = sequence_output[i][:l_i[0][1]]
            merged_cls.append(sequence_output[i][0])
            for j in range(1,n_s):
                gap = l_i[j-1][1]+l_i[j-1][0]-l_i[j][1]-l_i[j][0]
                if j==n_s-1:
                    output = torch.cat((output,sequence_output[i+j][gap+512:]),dim=0)

                else:
                    output = torch.cat((output,sequence_output[i+j][gap+511:511]),dim=0)
                merged_cls.append(sequence_output[i+j][0])
            weights = torch.tensor([1.0] + [0.8] * (len(merged_cls) - 2) + [1.0] if len(merged_cls) > 1 else [1.0],
                                   device=merged_cls[0].device).unsqueeze(1)
            weights = weights / weights.sum()
            final_cls = torch.sum(torch.stack(merged_cls) * weights, dim=0)
            cls_output.append(final_cls)
            output = F.pad(output,(0,0,0,c-l_seq))


            new_output.append(output)

            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        return sequence_output#, torch.stack(cls_output, dim=0)
