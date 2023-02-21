import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


#Embedding处理
class Embedding(nn.Module):
    def __init__(self,
                 vocab_size, #词汇长度
                 max_embedding_size, #embedding长度
                 segment_num, #片段数量
                 maxlen = 5000, #最大长度
                 ):
        super(Embedding, self).__init__()
        self.segment_num = segment_num
        self.max_embedding_size = max_embedding_size
        self.tok_embed = nn.Embedding(vocab_size, max_embedding_size)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, max_embedding_size)  # position embedding
        self.seg_embed = nn.Embedding(segment_num, max_embedding_size)  # segment(token type) embedding
        self.norm = nn.LayerNorm(max_embedding_size)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


#点乘相关度
class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 d_k, #数据维度
                 ):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


#前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,
                 max_embedding_size, #输入词矢量维度
                 hidden_layer_size, #隐藏层维度
                 dropout_prob, #dropout概率
                 hidden_act, #操作
                 ):
        super(PoswiseFeedForwardNet, self).__init__()
        self.hidden_act = hidden_act
        self.layerFirst = nn.Linear(max_embedding_size, hidden_layer_size)
        self.layerLast = nn.Linear(hidden_layer_size, max_embedding_size)
        self.norm = nn.LayerNorm(max_embedding_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        residual = inputs
        if self.hidden_act == 'gelu':
            return self.norm(residual + self.layerLast(self.dropout(F.relu(self.layerFirst(inputs)))))


#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 max_embedding_size, #输入词矢量维度
                 d_k, #k的维度
                 d_v, #v的维度
                 attention_head_num, #注意力头的数量
                 ):
        super(MultiHeadAttention, self).__init__()
        #多注意力机制主要分为转化、分头、计算注意力三步
        self.max_embedding_size = max_embedding_size
        self.attention_head_num = attention_head_num
        self.d_k = d_k
        self.d_v = d_v
        self.W_K = nn.Linear(max_embedding_size, d_k * attention_head_num)
        self.W_Q = nn.Linear(max_embedding_size, d_k * attention_head_num)
        self.W_V = nn.Linear(max_embedding_size, d_v * attention_head_num)

    def forward(self, Q, K, V, attn_mask):
        residual = Q
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.attention_head_num, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.attention_head_num, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.attention_head_num, self.d_v).transpose(1, 2)
        #注意力遮挡
        #重复attention头的数量的次数
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.attention_head_num, 1, 1)
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_head_num * self.d_v)
        #映射之后输出
        output = nn.Linear(self.attention_head_num * self.d_v, self.max_embedding_size)(context)
        return nn.LayerNorm(self.max_embedding_size)(output + residual)


#Encoder
class Encoder(nn.Module):
    def __init__(self,
                 max_embedding_size,  # 输入词矢量维度
                 hidden_layer_size,  # 隐藏层维度
                 dropout_prob,  # dropout概率
                 hidden_act,  # 操作
                 d_k,  # k的维度
                 d_v,  # v的维度
                 attention_head_num,  # 注意力头的数量
                 ):
        super(Encoder, self).__init__()
        #Encoder由一个多头注意力层一个前馈网络组成
        self.multiHeadAttention = \
            MultiHeadAttention(max_embedding_size,
                              d_k,
                              d_v,
                              attention_head_num)
        self.poswiseFeedForwardNet = \
            PoswiseFeedForwardNet(max_embedding_size,
                                  hidden_layer_size,
                                  dropout_prob,
                                  hidden_act)

    def forward(self, enc_inputs, enc_self_attn_mask):
        outputs = self.multiHeadAttention(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        outputs = self.poswiseFeedForwardNet(outputs)
        return outputs


#首先考虑Bert本身的构造
class Bert(nn.Module):
    def __init__(self,
                 vocab_size, #词单大小
                 max_embedding_size, #embedding之后的大小
                 layers_num, #encoder组数
                 segment_num, #片段数量
                 maxlen = 5000, #最大长度
                 ):
        super(Bert, self).__init__()
        self.embedding = \
            Embedding(vocab_size,
                      max_embedding_size,
                      segment_num,
                      maxlen)
        self.max_embedding_size = max_embedding_size
        self.layers_num = layers_num
        self.encoder = nn.ModuleList([Encoder() for _ in range(self.layers_num)])
        #用于cls类别输出
        self.clsFc = nn.Linear(max_embedding_size, max_embedding_size)
        #用于cls激活
        self.clsActive = nn.Tanh()
        #用于mlm的输出
        self.linear = nn.Linear(max_embedding_size, max_embedding_size)
        self.activ2 = F.gelu()
        self.norm = nn.LayerNorm(max_embedding_size)
        #cls分类器
        self.classifier = nn.Linear(max_embedding_size, 2)
        #mlm解码
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_pooled = self.activ1(self.fc(output[:, 0]))
        logits_clsf = self.classifier(h_pooled)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_clsf