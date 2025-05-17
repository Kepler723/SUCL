import torch
import torch.nn.functional as F
import numpy as np


def sliding_window_tokenize(input_id, start_token, end_token, window_size=510, overlap=255):
    input_slices = []
    l = len(input_id)
    if l <= window_size:
        return [torch.cat([start_token, input_id, end_token], dim=-1)], window_size - l
    else:
        i= 0
        while i < len(input_id)-overlap:
            start = i
            if i+window_size>len(input_id):
                start = len(input_id)- window_size
            input_slices.append(torch.cat([start_token,input_id[start:min(len(input_id),start+window_size)],end_token],dim=-1))
            i+=overlap
        return input_slices, window_size - (len(input_id) % overlap)


def slide_window(model, input_ids, attention_mask, start_tokens, end_tokens, device):

    max_len = input_ids.shape[1]
    input_ids = input_ids.to(device)

    attention_mask = attention_mask.to(device)
    start_tokens = torch.tensor(start_tokens).to(device)
    end_tokens = torch.tensor(end_tokens).to(device)

    seq_lens = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()

    input_slices = []
    mask_slices = []
    _len = []
    _gap = []
    _len_ = []
    _l_ = []
    for input_id, seq_len in zip(input_ids, seq_lens):
        input_id = input_id[:seq_len][1:][:-1]
        input_slice, last_lap = sliding_window_tokenize(input_id, start_tokens, end_tokens)
        _len_.append(len(input_slice))
        _l_.extend([len(slice) for slice in input_slice])
        mask_slice = [torch.tensor([1] * len(slice) + [0] * (512 - len(slice))).to(device) for slice in input_slice]

        input_slice = [F.pad(slice, (0, 512 - len(slice))) for slice in input_slice]
        input_slices.extend(input_slice)
        mask_slices.extend(mask_slice)
        _len.append(len(input_slice))
        _gap.append(last_lap)

    input_ids = torch.stack(input_slices, dim=0).to(device)
    attention_mask = torch.stack(mask_slices, dim=0).to(device)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )

    sequence_output, attention = output[0], output[-1][-1]
    new_output = []
    new_attention = []
    processed_output = []
    start = 0
    gap_offset = 0
    sequence_output = [output[mask.bool()] for output, mask in zip(sequence_output, mask_slices)]
    attention = [output[:, mask.bool(), :][:, :, mask.bool()] for output, mask in zip(attention, mask_slices)]
    for _l in _len:
        masked_output = sequence_output[start:start + _l]
        masked_attention = attention[start:start + _l]
        if _l == 1:
            result_output = masked_output[0]
            result_attention = masked_attention[0][:, 1:masked_attention[0].shape[1] - 1, 1:masked_attention[0].shape[1] - 1]
        else:
            result_output = masked_output[0][:len(masked_output[0]) - 1]
            result_attention = masked_attention[0][:, 1:masked_attention[0].shape[1] - 1, 1:masked_attention[0].shape[1] - 1]
            for index in range(1, _l):
                temp_output = masked_output[index][1:len(masked_output[0]) - 1]
                gap = 255 if index != _l - 1 else _gap[gap_offset]
                avg_gap = torch.mean(torch.stack((result_output[-gap:], temp_output[:gap])), dim=0)
                result_output = torch.cat((result_output[:-gap], avg_gap, temp_output[gap:]), dim=0)

                result_attention = torch.nn.functional.pad(result_attention, (0, 510-gap, 0, 510-gap))
                current_size = result_attention.shape[1]
                temp_attention = masked_attention[index][:, 1:masked_attention[index].shape[1] - 1, 1:masked_attention[0].shape[1] - 1]
                temp_size = temp_attention.shape[1]
                result_attention[:, current_size-temp_size:, current_size-temp_size:] += temp_attention
                result_attention[:, current_size - temp_size:, current_size - temp_size:] /= 2
            result_output = torch.cat((result_output, masked_output[-1][-1:]), dim=0)
        processed_output.append(len(result_output))
        new_output.append(F.pad(result_output, (0, 0, 0, max_len - len(result_output))))
        new_attention.append(result_attention)
        start += _l
        gap_offset += 1
    assert seq_lens == processed_output, f"seq_lens: {seq_lens} \n processed_output: {processed_output} \n gap: {_gap}"
    sequence_output = torch.stack(new_output, dim=0)

    return sequence_output, new_attention

