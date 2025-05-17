import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from cut_seq import process_long_input
from pyexpat import features
from torch.nn.utils.rnn import pad_sequence
from attention import MultiHeadAttention, EmbedLayer, EntityAttention, EntityMentionAggregation
from ContrastiveLoss import SupConLoss,SupConLossByKL
from slide_window import slide_window
import numpy as np

class DocREModel(nn.Module):
    def __init__(self, config, model, args):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.emb_size = 768
        self.block_size = 64
        self.num_labels = args.num_labels
        self.num_heads = 8
        self.attn_dropout = args.attn_dropout
        self.embedding_dim = 20
        self.linear_size = args.linear_size
        self.temperature = 0.05

        self.attention = args.attention
        self.device = args.device
        self.alpha = args.alpha
        self.beta = args.beta
        temp = args.es + args.ea + args.attention +1

        # self.learnable_params = LearnableHyperParams(init_alpha=args.alpha, init_beta=args.beta)
        self.entity_attention = EntityMentionAggregation(config.hidden_size)
        # self.src_attention = EntityMentionAggregation(config.hidden_size)
        # self.tgt_attention = EntityMentionAggregation(config.hidden_size)
        self.entity_extractor = nn.Linear(temp * config.hidden_size, self.emb_size)
        # self.head_extractor = nn.Linear(temp * config.hidden_size, self.emb_size)
        # self.tail_extractor = nn.Linear(temp * config.hidden_size, self.emb_size)
        self.mutiheadattention = MultiHeadAttention(self.emb_size, self.num_heads, self.attn_dropout)
        # self.mutiheadattention = MultiHeadAttention(self.emb_size, self.num_heads, self.attn_dropout)
        # self.mutiheadattention1 = MultiHeadAttention(self.emb_size, self.num_heads, self.attn_dropout)
        self.bilinear = nn.Linear(self.emb_size * self.block_size, config.num_labels)
        self.contrastiveLossByKL = SupConLossByKL(temperature=self.temperature)

        self.contrastiveLoss = SupConLoss(temperature=self.temperature)
        self.cl_linear = nn.Linear((1+args.ucl)*self.emb_size, self.linear_size)


    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        if self.attention==0:
            sequence_output = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
            return sequence_output
        else:
            sequence_output, attention = slide_window(self.model, input_ids, attention_mask, start_tokens, end_tokens, self.device)
            return sequence_output, attention

    def getSrcAndTgt(self, entity_poss, sequence_outputs):

        def pad_and_combine(tensors, pad_value=0):
            """
            填充并组合张量，同时返回掩码
            
            Args:
                tensors: 张量列表，每个形状为 [hidden_size, num_mentions]
                pad_value: 填充值
                
            Returns:
                output: 填充后的批次张量 [batch_size, hidden_size, max_mentions]
                masks: 有效提及的掩码 [batch_size, max_mentions]
            """
            # 找出最大的列数
            max_cols = max(tensor.shape[1] for tensor in tensors)

            # 初始化列表来保存处理后的tensor和掩码
            padded_tensors = []
            mention_masks = []

            # 对每个tensor进行处理
            for tensor in tensors:
                # 计算需要填充的列数
                padding_cols = max_cols - tensor.shape[1]

                # 创建一个填充tensor
                padding_tensor = torch.full((tensor.shape[0], padding_cols), pad_value).to(tensor.device)

                # 将输入tensor和填充tensor在列方向上进行合并，然后增加一个批次维度
                padded_tensor = torch.cat([tensor, padding_tensor], dim=1).unsqueeze(0)
                padded_tensors.append(padded_tensor)

                # 创建掩码: 1表示实际提及，0表示填充位置
                mask = torch.cat([
                    torch.ones(tensor.shape[1]),
                    torch.zeros(padding_cols)
                ]).unsqueeze(0).to(tensor.device)
                mention_masks.append(mask)

            # 使用 torch.cat 进行合并
            output = torch.cat(padded_tensors, dim=0)
            masks = torch.cat(mention_masks, dim=0)

            return output, masks

        # entity representations
        srcs = []
        tgts = []

        srcs_trans = []
        tgts_trans = []
        # entity_poss = [batch_size,2,n]
        # sequence_outputs = [batch_size,512,762]
        for entity_pos, sequence_output in zip(entity_poss, sequence_outputs):
            src, tgt = entity_pos[0], entity_pos[1]
            # print(tgt)
            # print(src,tgt)
            # print(sequence_output)
            src_mention_emb = []
            tgt_mention_emb = []
            for i in src:
                src_mention_emb.append(sequence_output[i])
            for i in tgt:
                tgt_mention_emb.append(sequence_output[i])
            # print(tgt_mention_emb)
            # mention_emb[num_embedding, hidden_size]
            src_mention_emb = torch.stack(src_mention_emb, dim=0).to(sequence_output)
            tgt_mention_emb = torch.stack(tgt_mention_emb, dim=0).to(sequence_output)

            # integrate mentions into entity[batch_size,hidden_size]
            # apply logsumexp to mentions' every column  to get the entity's corresponding column
            srcs.append(torch.logsumexp(src_mention_emb, dim=0).to(sequence_output))
            tgts.append(torch.logsumexp(tgt_mention_emb, dim=0).to(sequence_output))

            # mention_emb[hidden_size, num_embedding]
            src_mention_emb = src_mention_emb.transpose(0, 1)
            tgt_mention_emb = tgt_mention_emb.transpose(0, 1)

            # trans[batch_size, hidden_size, num_embedding]
            srcs_trans.append(src_mention_emb)
            tgts_trans.append(tgt_mention_emb)

        src_origin_e_emb = torch.stack(srcs, dim=0).to(sequence_outputs)
        tgt_origin_e_emb = torch.stack(tgts, dim=0).to(sequence_outputs)
        srcs_trans, src_masks = pad_and_combine(srcs_trans, pad_value=0)
        tgts_trans, tgt_masks = pad_and_combine(tgts_trans, pad_value=0)
        # [batch_size, num_embedding, hidden_size]
        srcs_trans = srcs_trans.transpose(1, 2)
        tgts_trans = tgts_trans.transpose(1, 2)

        return src_origin_e_emb, tgt_origin_e_emb, srcs_trans, tgts_trans, src_masks, tgt_masks  # 实体表征，实体表征，提及表征集合，提及表征集合

    def getSentEmb(self, mention_pos, sent_pos, sequence_output):

        for sent in sent_pos:
            if mention_pos < sent:
                return sequence_output[sent]
        # if len(sent_pos)==0 or sent_pos[0]>len(sequence_output):
        #     print(sent_pos)
        #     print(sequence_output)
        return sequence_output[sent_pos[0]]

    def get_Mention_sent_emb(self, entity_poss, sent_poss, sequence_outputs):

        srcSentEmbeddings, tgtSentEmbeddings = [], []

        for entity_pos, sent_poss, sequence_output in zip(entity_poss, sent_poss, sequence_outputs):
            src = entity_pos[0]
            tgt = entity_pos[1]
            srcSentEmbedding, tgtSentEmbedding = [], []
            for i in src:
                srcSentEmbedding.append(self.getSentEmb(i, sent_poss, sequence_output))
            for i in tgt:
                tgtSentEmbedding.append(self.getSentEmb(i, sent_poss, sequence_output))
            srcSentEmbedding = torch.stack(srcSentEmbedding, dim=0).to(sequence_output)
            tgtSentEmbedding = torch.stack(tgtSentEmbedding, dim=0).to(sequence_output)
            # [batch_size, num_embedding, hidden_size]
            srcSentEmbeddings.append(srcSentEmbedding)
            tgtSentEmbeddings.append(tgtSentEmbedding)
        # 将他们组合起来，并padding，得到最终的
        srcSentEmbeddings = pad_sequence(srcSentEmbeddings, batch_first=True, padding_value=0)
        tgtSentEmbeddings = pad_sequence(tgtSentEmbeddings, batch_first=True, padding_value=0)
        return srcSentEmbeddings, tgtSentEmbeddings


    def get_logits(self, config, input_ids=None, attention_mask=None, entity_pos=None, sent_pos=None, src_type=None, tgt_type=None):
        sequence_output= self.encode(input_ids, attention_mask)

        clsEmbedding = sequence_output[:, 0, :]
        # ERA
        if config.es:
            srcs_e, tgts_e, srcs_ms, tgts_ms, src_masks, tgt_masks = self.getSrcAndTgt(entity_pos, sequence_output)  # 实体的原始表征
            src_entities = self.entity_attention(srcs_ms, tgts_ms, src_masks, tgt_masks)  # 实体的提及表征
            tgt_entities = self.entity_attention(tgts_ms, srcs_ms, tgt_masks, src_masks)  # 实体的提及表征
        if config.ea:
            srcSentEmbeddings, tgtSentEmbeddings = self.get_Mention_sent_emb(entity_pos, sent_pos, sequence_output)
            srcEmb, _ = self.mutiheadattention(tgt_entities, srcSentEmbeddings, srcs_ms)  # 使用多头聚合实体提及后的实体表征
            tgtEmb, _ = self.mutiheadattention(src_entities, tgtSentEmbeddings, tgts_ms)

            # dist = torch.tensor(dist)
            # disEmb = self.dist_embed_dir(dist.to(sequence_output).int())#实体间的距离嵌入

            srcs_classify = torch.tanh(self.entity_extractor(torch.cat([src_entities, srcEmb, clsEmbedding], dim=1)))
            tgts_classify = torch.tanh(self.entity_extractor(torch.cat([tgt_entities, tgtEmb, clsEmbedding], dim=1)))
        elif config.es:
            srcs_classify = torch.tanh(self.entity_extractor(torch.cat([src_entities,clsEmbedding], dim=1)))
            tgts_classify = torch.tanh(self.entity_extractor(torch.cat([tgt_entities,clsEmbedding], dim=1)))
        else:
            srcs_classify = torch.tanh(self.entity_extractor(torch.cat([clsEmbedding], dim=1)))
            tgts_classify = torch.tanh(self.entity_extractor(torch.cat([clsEmbedding], dim=1)))

        b1 = srcs_classify.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = tgts_classify.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

        return srcs_classify, tgts_classify, self.bilinear(bl)
    def forward(self, config, input_ids=None, attention_mask=None,  entity_pos=None,
                sent_pos=None, mask_input_ids=None, mask_attention_mask=None, mask_entity_pos=None, mask_sent_pos=None,
                labels=None, src_type=None, tgt_type=None, dist=None, hts=None, entity_type=None, instance_mask=None, sent_window = None
                ):

        srcs_classify, tgts_classify, logits = self.get_logits(config, input_ids, attention_mask, entity_pos, sent_pos, src_type, tgt_type)
        # mask_srcs_classify, mask_tgts_classify, mask_logits = self.get_logits(config, mask_input_ids, mask_attention_mask, mask_entity_pos, mask_sent_pos)

        if labels is not None:

            labels = torch.tensor(labels).to(logits)
            labels_index = torch.argmax(labels, dim=1)
            loss = nn.CrossEntropyLoss()(logits.float(), labels_index)

            # src_type, tgt_type = torch.tensor(src_type).to(logits), torch.tensor(tgt_type).to(logits)
            # src_index = torch.argmax(src_type, dim=1)
            # tgt_index = torch.argmax(tgt_type, dim=1)
            # src_type_loss = nn.CrossEntropyLoss()(pred_src_type.float(), src_index)
            # tgt_type_loss = nn.CrossEntropyLoss()(pred_tgt_type.float(), tgt_index)
            # loss += 0.1*src_type_loss + 0.1*tgt_type_loss
            # loss += self.ucl(logits, mask_logits)

            if config.ucl:
                srcs_classify2, tgt_classify2, logits2, = self.get_logits(config, input_ids, attention_mask, entity_pos,sent_pos)
                loss2 = nn.CrossEntropyLoss()(logits2.float(), labels_index)
                loss += loss2 + self.ucl(logits, logits2)

            if config.scl:
                if config.ucl:
                    merged_logits = torch.cat([logits, logits2], dim=0)
                    merged_labels = torch.cat([labels, labels], dim=0)
                    merged_srcs_classify = torch.cat([srcs_classify, srcs_classify2], dim=0)
                    merged_tgts_classify = torch.cat([tgts_classify, tgt_classify2], dim=0)
                    loss += self.scl(config.scl, merged_labels, merged_logits,
                                     merged_srcs_classify, merged_tgts_classify)
                else:
                    loss += self.scl(config.scl, labels, logits, srcs_classify, tgts_classify)
            return loss

        probs = F.softmax(logits, dim=1)

        rows = torch.arange(logits.size(0))
        preds_index = torch.argmax(probs, dim=1)
        preds = torch.zeros_like(logits)
        preds[rows, preds_index] = 1

        return preds

    def scl(self, scl_type, labels, logits=None, src_classify=None, tgt_classify=None):
        if scl_type:
            closs = self.contrastiveLossByKL(logits=logits, labels=labels)
        else:
            cl_feature = torch.tanh(
                self.cl_linear_ucl(torch.cat([src_classify, tgt_classify], dim=1), dim=0))
            closs = self.contrastiveLoss(features=cl_feature, labels=labels)
        return self.beta * closs

    def ucl(self, logits1, logits2):
        p = torch.log_softmax(logits1, dim=-1)
        q = torch.log_softmax(logits2, dim=-1)

        p_tec = torch.softmax(logits1, dim=-1)
        q_tec = torch.softmax(logits2, dim=-1)

        kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
        reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()

        return self.alpha * (kl_loss + reverse_kl_loss)

    def cal_ET_loss(self, batch_ET_reps, batch_epair_types):

        batch_epair_types = batch_epair_types.T.flatten()
        batch_ET_loss = F.cross_entropy(batch_ET_reps, batch_epair_types)

        return batch_ET_loss