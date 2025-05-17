import torch
from torch import nn
import torch.nn.functional as F
import math

class EntityMentionAggregation(nn.Module):
    def __init__(self, hidden_size):
        super(EntityMentionAggregation, self).__init__()
        self.hidden_size = hidden_size

        # 自注意力部分
        self.self_query = nn.Linear(hidden_size, hidden_size)
        self.self_key = nn.Linear(hidden_size, hidden_size)
        self.self_value = nn.Linear(hidden_size, hidden_size)

        # 交叉注意力部分
        self.cross_query = nn.Linear(hidden_size, hidden_size)
        self.cross_key = nn.Linear(hidden_size, hidden_size)

        # 融合门控机制
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, head_mentions, tail_mentions, head_mask=None, tail_mask=None):
        """
        输入:
            head_mentions: [batch_size, max_head_mentions, hidden_size]
            tail_mentions: [batch_size, max_tail_mentions, hidden_size]
            head_mask: [batch_size, max_head_mentions]
            tail_mask: [batch_size, max_tail_mentions]
        """
        batch_size = head_mentions.size(0)

        # 1. 自注意力机制
        self_q = self.self_query(head_mentions)
        self_k = self.self_key(head_mentions)
        self_v = self.self_value(head_mentions)

        # 计算自注意力分数
        self_attn_scores = torch.bmm(self_q, self_k.transpose(1, 2)) / math.sqrt(self.hidden_size)
        if head_mask is not None:
            self_attn_scores = self_attn_scores.masked_fill(
                head_mask.unsqueeze(1) == 0, -65504)
        self_attn_weights = F.softmax(self_attn_scores, dim=-1)
        self_attn_output = torch.bmm(self_attn_weights, self_v)

        # 2. 交叉注意力机制
        cross_q = self.cross_query(head_mentions)
        cross_k = self.cross_key(tail_mentions)

        # 计算交叉注意力分数
        cross_attn_scores = torch.bmm(cross_q, cross_k.transpose(1, 2)) / math.sqrt(self.hidden_size)
        if tail_mask is not None:
            cross_attn_scores = cross_attn_scores.masked_fill(
                tail_mask.unsqueeze(1) == 0, -65504)
        cross_attn_weights = F.softmax(cross_attn_scores, dim=-1)
        cross_attn_output = torch.bmm(cross_attn_weights, tail_mentions)

        # 3. 融合两种注意力结果
        gate_input = torch.cat([self_attn_output, cross_attn_output], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        fused_output = gate * self_attn_output + (1 - gate) * cross_attn_output

        # 4. 聚合所有提及
        if head_mask is not None:
            mention_weights = F.softmax(head_mask.float().unsqueeze(-1), dim=1)
            entity_repr = torch.sum(fused_output * mention_weights, dim=1)
        else:
            entity_repr = torch.mean(fused_output, dim=1)

        return entity_repr
class EntityAttention(nn.Module):
    def __init__(self, hidden_size):
        super(EntityAttention, self).__init__()
        self.query_transform = nn.Linear(hidden_size, hidden_size)
        self.key_transform = nn.Linear(hidden_size, hidden_size)
        self.value_transform = nn.Linear(hidden_size, hidden_size) 
        
    def forward(self, src_mentions, tgt_mentions, src_mask=None, tgt_mask=None):
        """
        批处理的提及对提及注意力计算
        
        Args:
            src_mentions: 源实体的提及 [batch_size, num_src_mentions, hidden_size]
            tgt_mentions: 目标实体的提及 [batch_size, num_tgt_mentions, hidden_size]
            src_mask: 源提及掩码 [batch_size, num_src_mentions]
            tgt_mask: 目标提及掩码 [batch_size, num_tgt_mentions]
        
        Returns:
            context: 注意力聚合结果 [batch_size, num_src_mentions, hidden_size]
            attention_weights: 注意力权重 [batch_size, num_src_mentions, num_tgt_mentions]
        """
        # 转换src提及为查询
        query = self.query_transform(src_mentions)  # [batch_size, num_src_mentions, hidden_size]
        
        # 转换tgt提及为键和值
        keys = self.key_transform(tgt_mentions)     # [batch_size, num_tgt_mentions, hidden_size]
        values = self.value_transform(tgt_mentions) # [batch_size, num_tgt_mentions, hidden_size]
        
        # 批量注意力计算 - 每个实例独立计算
        attention_scores = torch.bmm(query, keys.transpose(1, 2))  # [batch_size, num_src_mentions, num_tgt_mentions]
        
        # 应用掩码(如果提供)
        # 应用掩码(如果提供)
        if tgt_mask is not None:
            # 确保掩码在正确设备上
            tgt_mask = tgt_mask.to(src_mentions.device)
            # 扩展掩码与注意力分数形状匹配
            mask = tgt_mask.unsqueeze(1).expand(-1, query.size(1), -1)
            
            # 使用与张量相同设备的填充值
            fill_value = torch.tensor(torch.finfo(attention_scores.dtype).min, device=attention_scores.device)
            attention_scores = attention_scores.masked_fill(mask == 0, fill_value)
        
        # 对每个源提及计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=2)  # [batch_size, num_src_mentions, num_tgt_mentions]
        
        # 加权聚合目标提及
        context = torch.bmm(attention_weights, values)  # [batch_size, num_src_mentions, hidden_size]
        
        return self.aggregate_mentions(context, src_mask, method="attention")

    def aggregate_mentions(self, context_vectors, mention_mask=None, method="mean"):
        """
        将提及级表示聚合为实体级表示
        """
        device = context_vectors.device  # 获取输入张量的设备
        dtype = context_vectors.dtype    # 获取输入张量的数据类型
        
        # 根据数据类型安全地选择填充值
        if dtype == torch.float16:
            safe_fill_value = -65504.0  # 半精度浮点的安全负值
        else:
            safe_fill_value = -1e9
        
        if method == "mean":
            # 平均池化聚合
            if mention_mask is not None:
                # 确保掩码在正确设备上
                mention_mask = mention_mask.to(device)
                mask_expanded = mention_mask.unsqueeze(2).float()
                sum_vectors = torch.sum(context_vectors * mask_expanded, dim=1)
                # 计算有效提及数(避免除零)
                mention_counts = torch.clamp(torch.sum(mention_mask, dim=1, keepdim=True), min=1.0)
                entity_repr = sum_vectors / mention_counts
            else:
                entity_repr = torch.mean(context_vectors, dim=1)
                
        elif method == "max":
            # 最大池化聚合
            if mention_mask is not None:
                mention_mask = mention_mask.to(device)
                mask_expanded = mention_mask.unsqueeze(2).float()
                # 使用安全填充值
                fill_value = torch.tensor(safe_fill_value, device=device, dtype=dtype)
                masked_vectors = context_vectors * mask_expanded + (1 - mask_expanded) * fill_value
                entity_repr = torch.max(masked_vectors, dim=1)[0]
            else:
                entity_repr = torch.max(context_vectors, dim=1)[0]
                
        elif method == "attention":
            # 注意力加权聚合
            if not hasattr(self, 'mention_query'):
                # 延迟初始化可学习的查询向量 - 确保在正确的设备上
                hidden_size = context_vectors.size(-1)
                self.mention_query = nn.Parameter(torch.randn(1, 1, hidden_size, device=device, dtype=dtype))
                self.mention_attn = nn.Linear(hidden_size, 1)
                # 将线性层移至正确设备
                self.mention_attn = self.mention_attn.to(device)
                        
            # 计算每个提及的重要性得分
            mention_scores = self.mention_attn(context_vectors)
                    
            # 应用掩码
            if mention_mask is not None:
                mention_mask = mention_mask.to(device)
                # 使用安全填充值
                fill_value = torch.tensor(safe_fill_value, device=device, dtype=mention_scores.dtype)
                mention_scores = mention_scores.masked_fill(mention_mask.unsqueeze(2) == 0, fill_value)
                        
            # 计算注意力权重并聚合
            mention_weights = F.softmax(mention_scores, dim=1)
            entity_repr = torch.sum(context_vectors * mention_weights, dim=1)
            
        return entity_repr

class ScaleDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q: query [b, l_q, d_q]
        :param k: keys [b, l_k, d_k]
        :param v: values [b, l_v, d_v]， k=v
        :param scale:
        :param attn_mask: masking  [b, l_q, l_k]
        :return:
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -1e12)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        # torch.nn.init.eye_(self.linear_k.weight)
        # torch.nn.init.eye_(self.linear_v.weight)
        # torch.nn.init.eye_(self.linear_q.weight)

        self.dot_product_attention = ScaleDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        entity_size = query.size(1)
        # [batch_size,hidden_size]
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        # print('before',key.shape,value.shape,query.shape)
        # split by heads
        key = key.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads, -1,
                                                                                           dim_per_head)
        value = value.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads,
                                                                                               -1, dim_per_head)
        query = query.reshape(batch_size, -1, num_heads, dim_per_head).transpose(1, 2).reshape(batch_size * num_heads,
                                                                                               -1, dim_per_head)
        # print('after',key.shape,value.shape,query.shape)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
            # scaled dot product attention
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.reshape(batch_size, num_heads, -1, dim_per_head).transpose(1, 2).reshape(batch_size, -1,
                                                                                                   num_heads * dim_per_head)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        # output = output.reshape(batch_size, entity_size, entity_size, -1 )
        output = output.reshape(batch_size, entity_size)
        output = self.layer_norm(output)

        return output, attention


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False, pretrained=None,
                 mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze
        self.ignore = ignore

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)
        self.embedding.weight.requires_grad = not freeze

        if pretrained:
            self.load_pretrained(pretrained, mapping)

        self.drop = nn.Dropout(dropout)

    def load_pretrained(self, pretrained, mapping):
        """
        Args:
            weights: (dict) keys are words, values are vectors
            mapping: (dict) keys are words, values are unique ids
            trainable: (bool)

        Returns: updates the embedding matrix with pre-trained embeddings
        """
        # if self.freeze:
        pret_embeds = torch.zeros((self.num_embeddings, self.embedding_dim))
        # else:
        # pret_embeds = nn.init.normal_(torch.empty((self.num_embeddings, self.embedding_dim)))
        for word in mapping.keys():
            if word in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word])
            elif word.lower() in pretrained:
                pret_embeds[mapping[word], :] = torch.from_numpy(pretrained[word.lower()])
        self.embedding = self.embedding.from_pretrained(pret_embeds, freeze=self.freeze)  # , padding_idx=self.ignore

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        for i in xs:
            if i > 20:
                print(True)
                print(i)
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)

        return embeds
