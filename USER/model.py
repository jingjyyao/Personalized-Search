"""
    author: jingyao
    date: 20210203
    the USER model, including 4 components: text encoder, session encoder, history encoder, unified task framework
"""

import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Layers import EncoderLayer, DecoderLayer


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table according to sin/cos '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # !!!注意：在我们的模型中，query下点击的document只和对应的query以及其他document进行attention
    # 这里采用先搭建图，再又图得出邻接矩阵的方式来生成mask
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk


class Encoder(nn.Module):
    """ 
        A transformer encoder layer 
        len_max_seq: max length of the input sequence
        embed_dim: embed dimension of the input
        d_model: dimension of input, in the first layer, equal to embed_dim
        d_inner: 
        n_layers: 
    """
    def __init__(self, len_max_seq, embed_dim, d_model, d_inner, n_layers, n_head,
        d_k, d_v, dropout=0.1):
        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, embed_dim, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_emb, src_pos, atten_mask=None, return_attns=False, needpos=False):
        enc_slf_attn_list = []

        if atten_mask == None:
            slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        else:
            slf_attn_mask = atten_mask.eq(1)  # attention!!! encoder内部的attention在mask时，是将值为1的位置，置成-inf，所以这里得用1减去，反一下
        
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        if needpos:
            enc_output = src_emb + self.position_enc(src_pos)
        else:
            enc_output = src_emb

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


def kernel_mus(n_kernels):
        l_mu = [1]
        if n_kernels == 1:
            return l_mu
        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma
    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma


class knrm(nn.Module):
    def __init__(self, k):
        super(knrm, self).__init__()
        tensor_mu = torch.FloatTensor(kernel_mus(k)).cuda()
        tensor_sigma = torch.FloatTensor(kernel_sigmas(k)).cuda()
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, k)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, k)
        self.dense = nn.Linear(k, 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1) # n*m*d*1
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)#soft-TF
        return log_pooling_sum

    def forward(self, inputs_q, inputs_d, mask_q, mask_d):
        q_embed_norm = F.normalize(inputs_q, 2, 2)
        d_embed_norm = F.normalize(inputs_d, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        output = F.tanh(self.dense(log_pooling_sum))
        return output

class CoAttention(nn.Module):
    def __init__(self, embed_dim=100, latent_dim=100, max_s_len=50):
        super(CoAttention, self).__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_s_len = max_s_len

        self.W1 = torch.rand((self.embed_dim, self.embed_dim), requires_grad=True).cuda()
        self.Wq = torch.rand((self.latent_dim, self.embed_dim), requires_grad=True).cuda()
        self.Wd = torch.rand((self.latent_dim, self.embed_dim), requires_grad=True).cuda()

        self.whq = torch.rand((1, self.latent_dim), requires_grad=True).cuda()
        self.whd = torch.rand((1, self.latent_dim), requires_grad=True).cuda()
        self.qd_linear = nn.Linear(2*self.latent_dim, self.embed_dim)

    def forward(self, query, doc): # query, doc: [batchsize, max_s_len, embed_dim]
        query_trans = query.transpose(2, 1)
        doc_trans = doc.transpose(2, 1)
        L = torch.tanh(torch.matmul(torch.matmul(query, self.W1), doc_trans)) # 行是q->d，列是d->q, QWD_T
        L_trans = L.transpose(2, 1) # DWQ_T

        Hq = torch.tanh(torch.matmul(self.Wq, query_trans) + torch.matmul(torch.matmul(self.Wd, doc_trans), L_trans)) # [batchsize, latend_dim, max_s_len]
        Hd = torch.tanh(torch.matmul(self.Wd, doc_trans) + torch.matmul(torch.matmul(self.Wq, query_trans), L))

        Aq = F.softmax(torch.matmul(self.whq, Hq)) # [batchsize, 1, max_s_len]
        Ad = F.softmax(torch.matmul(self.whd, Hd)) # [batchsize, 1, max_s_len]

        coAttn_q = torch.matmul(Aq, query).squeeze(1)
        coAttn_d = torch.matmul(Ad, doc).squeeze(1)

        coAttn_qd = torch.cat([coAttn_q, coAttn_d], dim=1)  # [batchsize, 2*latent_dim]
        return self.qd_linear(coAttn_qd)

class Simple_CoAttention(nn.Module):
    def __init__(self, embed_dim=100, latent_dim=100, max_s_len=50):
        super(Simple_CoAttention, self).__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_s_len = max_s_len

        self.W1 = torch.rand((self.embed_dim, self.embed_dim), requires_grad=True).cuda()

        self.qd_linear = nn.Linear(2*self.embed_dim, self.embed_dim)

    def forward(self, query, doc): # query, doc: [batchsize, max_s_len, embed_dim]
        # norm_query = F.normalize(query, 2, 2)
        # norm_doc = F.normalize(doc, 2, 2)
        # norm_query_trans = norm_query.transpose(2, 1)
        # norm_doc_trans = norm_doc.transpose(2, 1)
        doc_trans = doc.transpose(2, 1)
        # L = torch.matmul(norm_query, norm_doc_trans)  # [batchsize, max_s_len, max_s_len] 行是q->d，列是d->q, QWD_T
        L = torch.tanh(torch.matmul(torch.matmul(query, self.W1), doc_trans))
        L_trans = L.transpose(2, 1) # DWQ_T

        Aq = F.softmax(torch.sum(L_trans, dim=1).unsqueeze(1), -1) # [batchsize, 1, max_s_len]
        Ad = F.softmax(torch.sum(L, dim=1).unsqueeze(1), -1) # [batchsize, 1, max_s_len]

        coAttn_q = torch.matmul(Aq, query).squeeze(1)
        coAttn_d = torch.matmul(Ad, doc).squeeze(1)

        coAttn_qd = torch.cat([coAttn_q, coAttn_d], dim=1)  # [batchsize, 2*embed_dim]
        return self.qd_linear(coAttn_qd)

class VanillaAtttention(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=100):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.Tanh())
        self.proj_v = nn.Linear(self.hidden_dim, 1)

    def forward(self, k, v, mask):
        """
        Args:
            k, v (tensor): [B, seq_len, input_dim]
            mask: [B, seq_len]

        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        weights = self.proj_v(self.proj(k)).squeeze(-1)  # [B, seq_len]
        if mask is not None:
            weights = weights.masked_fill(mask.eq(0), -np.inf) # 注意这里是为1赋值-inf
        weights = torch.softmax(weights, dim=-1) # [B, seq_len]
        return torch.bmm(weights.unsqueeze(1), v).squeeze(1), weights # [B, 1, seq_len], [B, seq_len, dim]



class Contextual(nn.Module):
    def __init__(self, max_s_len, max_sess_len, max_sess_num, max_query_num, embed_dim, batch_size, embed_path, vocab_path, user_path, 
        source_path, source_embed_path, d_model=100, d_inner=2048, n_layers=1, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        # configuations
        self.max_s_len = max_s_len
        self.max_sess_len = max_sess_len  # number of queries in a session
        self.max_sess_num = max_sess_num  # number of sessions in the whole history
        self.max_query_num = max_query_num  # number of clicked queries in the whole history
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.vocabulary = pickle.load(open(vocab_path, 'rb'))
        self.userdict = pickle.load(open(user_path, 'rb'))
        self.source2id = pickle.load(open(source_path, 'rb'))

        # initialize embeddings
        self.embedding = nn.Embedding(len(self.vocabulary)+1, self.embed_dim)
        self.embedding.weight.data.copy_(self.load_embedding(embed_path))

        self.user_embedding = nn.Embedding(len(self.userdict)+1, self.embed_dim)

        self.type_embedding = nn.Embedding(4, self.embed_dim) # 3 types of nodes: recommend, search, document
        
        self.source_embedding = nn.Embedding(len(self.source2id)+1, self.embed_dim) # different sources of browsed/clicked documents
        self.source_embedding.weight.data.copy_(self.load_source_embedding(source_embed_path))

        self.knrm = knrm(11)

        self.word_attention = VanillaAtttention(input_dim=d_model, hidden_dim=d_model)
        self.coattention = Simple_CoAttention(embed_dim=self.embed_dim, latent_dim=self.embed_dim, max_s_len=self.max_s_len)

        self.encoder_query = Encoder(len_max_seq=max_s_len, embed_dim=embed_dim, d_model=d_model,
            d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.encoder_session = Encoder(len_max_seq=max_sess_len, embed_dim=embed_dim, d_model=d_model,
            d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.encoder_history = Encoder(len_max_seq=max_sess_len*max_sess_num+1, embed_dim=embed_dim, d_model=d_model,
            d_inner=d_inner, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.feature_layer = nn.Sequential(nn.Linear(122, 1),nn.Tanh())
        self.joint_score_layer = nn.Sequential(nn.Linear(126, 1), nn.Tanh())
        self.search_score_layer = nn.Sequential(nn.Linear(125, 1), nn.Tanh())
        self.recommend_score_layer = nn.Linear(2, 1)
        self.criterion = nn.CrossEntropyLoss()


    def load_source_embedding(self, source_embed_path): # load the pretrained word embedding
        weight = torch.zeros(len(self.source2id)+1, self.embed_dim)
        with open(source_embed_path, 'r') as fr:
            for line in fr:
                line = line.strip().split(" ")
                wordid = self.source2id[line[0]]
                weight[wordid, :] = torch.FloatTensor([float(t) for t in line[1:]]) 
        print("Successfully load the source vectors...")
        return weight


    def load_embedding(self, embed_path): # load the pretrained word embedding
        weight = torch.zeros(len(self.vocabulary)+1, self.embed_dim)
        weight[-1] = torch.rand(self.embed_dim)
        with open(embed_path, 'r') as fr:
            for line in fr:
                line = line.strip().split(" ")
                wordid = self.vocabulary[line[0]]
                weight[wordid, :] = torch.FloatTensor([float(t) for t in line[1:]]) 
        print("Successfully load the word vectors...")
        return weight


    def pairwise_loss(self, score1, score2):
        return (1/(1+torch.exp(score2-score1)))


    def type_mask(self, sample_type, target = 'recommend'):  # True for recommend sample, False for search sample
        if target == 'recommend':
            return sample_type.eq(0).type(torch.float).unsqueeze(-1)  # 0:recommend, 1:search
        else:
            return sample_type.eq(1).type(torch.float).unsqueeze(-1)  # 0:recommend, 1:search


    def attention(self, q, k, v, mask=None):
        """ compute weighted sum, with attention
            q: [1, embed_dim],
            k: [batchs                                                                                                                                                                                                                                                                                                                                                                    ize, max_s_len, embed_dim]
        """
        attn = torch.matmul(q, k.transpose(1, 2))  # [batchsize, 1, max_s_len]
        if mask is not None:
            attn = attn.masked_fill(mask.eq(0).unsqueeze(1), -np.inf) # 注意这里是为1赋值-inf

        attn = F.softmax(attn, dim=2)
        output = torch.bmm(attn, v).squeeze(1)  # [batchsize, emb_dim]
        return output


    def generate_sess_pos(self, long_pos):
        batch_size = long_pos.shape[0]
        long_sess_pos = torch.zeros((batch_size, self.max_sess_num, self.max_sess_len), dtype=torch.long)
        for i in range(batch_size):
            for j in range(self.max_sess_num):
                for k in range(self.max_sess_len):
                    if long_pos[i,j,k] != 0:
                        long_sess_pos[i,j,k] = j+1
        return long_sess_pos


    def forward(self, sample_type, user, query, doc, doc_source, feature, delta, label,
        short_history, short_types, short_sources, short_pos, long_history, long_types, long_sources, long_pos,
        history_qd, query_pos):
        """
            Args:
                sample_type  标记每个样本是针对搜索/推荐任务 [batch]
                user  每个样本对应的用户id [batch]
                doc1,doc2  正例/负例文档 [batch, s_len]
                doc1_source,doc2_source  正例/负例文档的来源 [batch]
                feature1/feature2  正例/负例文档对应的SLTB特征 [batch, feat_size]
                label  标签，指向正例文档 [batch]
                short_history  当前session历史中的用户行为 [batch, sess_len, s_len]
                short_types  当前session历史中用户行为的类别，搜索/推荐 [batch, sess_len]
                short_sources  当前session历史中每个行为下文档的来源 公众号/搜索百科... [batch, sess_len]
                short_pos  当前session历史中行为的位置 [batch, sess_len]
                long_history  历史session中的用户行为 [batch, sess_num, sess_len, s_len]
                long_types  历史session中用户行为的类别，搜索/推荐 [batch, sess_num, sess_len]
                long_sources  历史session中每个行为下文档的来源 公众号/搜索百科... [batch, sess_num, sess_len]
                long_pos  历史session中行为的位置 [batch, sess_num, sess_len]
                history_qd  历史中所有包含点击文档的查询，以及对应的点击文档 [batch, max_query, 2, sess_len]
                query_pos  历史包含点击的查询在历史序列中的位置 [batch, max_query]
        """
        ### Step1: process behavior history for both search & recommendation in the same way
        # fuse the information of the query and clicked documents to get the representation for historical query node, with co-attention
        history_q_embed = self.embedding(history_qd[:,:,0,:]).view(-1, self.max_s_len, self.embed_dim)
        history_d_embed = self.embedding(history_qd[:,:,1,:]).view(-1, self.max_s_len, self.embed_dim) # [batchsize*max_query_num, max_s_len, embed_dim] # 不足max_query_num的查询都用最后一个query补齐
        history_q_encode_1, *_ = self.encoder_query(history_q_embed, history_qd[:,:,0,:].view(-1, self.max_s_len))
        history_d_encode_1, *_ = self.encoder_query(history_d_embed, history_qd[:,:,1,:].view(-1, self.max_s_len))
        history_coAttn_qd = torch.reshape(self.coattention(history_q_encode_1, history_d_encode_1), (-1, self.max_query_num, self.embed_dim)) # [batchsize, max_query_num, embed_dim]

        # query-level transformer, self-attention within each query/document
        cand_num = doc.shape[1]
        all_history = torch.cat([long_history, short_history], 1) # [batchsize, max_sess_len*max_sess_num+max_sess_len-1, max_s_len]
        all_history_mask = all_history.view(-1, self.max_s_len)

        # embedding layer
        query_embed = self.embedding(query)  # [batchsize, max_s_len, emb_dim]
        doc_embed = self.embedding(doc).view(-1, self.max_s_len, self.embed_dim)  # [batchsize, npratio+1, max_s_len, emb_dim]
        all_history_embed = self.embedding(all_history).view(-1, self.max_s_len, self.embed_dim) # [batchsize*(max_sess_len*max_sess_num+max_sess_len-1), max_s_len, embed_dim]

        all_history_encode_1, *_ = self.encoder_query(all_history_embed, all_history_mask) # encode_1 表示第一层transformer的输出  [batchsize*(max_sess_len*max_sess_num+max_sess_len-1), max_s_len, embed_dim]

        query_encode_1, *_ = self.encoder_query(query_embed, query)  # [batchsize, max_s_len, embed_dim]
        doc = doc.view(-1, self.max_s_len)
        doc_encode_1, *_ = self.encoder_query(doc_embed, doc) # [batchsize*(npratio+1), max_s_len, emb_dim]

        # sum up all word representation in a query/doc, weights are calculated based on a word attention
        all_history_encode_1, _ = self.word_attention(all_history_encode_1, all_history_encode_1, all_history_mask)
        all_history_encode_1 = all_history_encode_1.view(-1, self.max_sess_num*self.max_sess_len+self.max_sess_len-1, self.embed_dim)

        doc_encode_1_sum, _ = self.word_attention(doc_encode_1, doc_encode_1, doc) # [batchsize, npratio+1, embed_dim]
        query_encode_1_sum, _ = self.word_attention(query_encode_1, query_encode_1, query)

        # set the representation of the query nodes as the history coAttn_qd
        query_pos = query_pos.unsqueeze(2).repeat(1, 1, self.embed_dim)
        all_history_encode_1.scatter_(dim=1, index=query_pos, src=history_coAttn_qd)
        long_encode_1, short_encode_1 = torch.split(all_history_encode_1, [self.max_sess_num*self.max_sess_len, self.max_sess_len-1], 1)

        # for recommendation, current query is initialized with user embedding
        user_embed = self.user_embedding(user)  # [batchsize, embed_dim]
        query_encode_1_sum = query_encode_1_sum*self.type_mask(sample_type, target='search') + user_embed*self.type_mask(sample_type, target='recommend')  # [batchsize, embed_dim]


        # session-level transformer, short history
        short_plus_query = torch.cat([short_encode_1, query_encode_1_sum.unsqueeze(1)], 1)  # [batchsize, max_sess_len, embed_dim]
        short_types_embed = self.type_embedding(short_types)  # [batchsize, max_sess_len, embed_dim]
        short_query_encode_2, *_ = self.encoder_session(short_plus_query, short_pos, needpos=False)
        short_query_encode_2, *_ = self.encoder_session(short_query_encode_2, short_pos, needpos=False)  # 当前session通过两层transformer，第一层捕捉relatedness，第二层fuse info
        short_encode_2, query_short_encode_2 = torch.split(short_query_encode_2, [self.max_sess_len-1, 1], 1)


        # session-level transformer, long history
        long_encode_1 = torch.reshape(long_encode_1, (-1, self.max_sess_len, self.embed_dim)) # [batchsize*sess_num, max_sess_len, embed_dim]
        long_types_embed = self.type_embedding(long_types)
        long_types_embed = long_types_embed.view(-1, self.max_sess_len, self.embed_dim)
        long_sess_pos = self.generate_sess_pos(long_pos) # [batch, sess_num, sess_len]
        long_encode_2, *_ = self.encoder_session(long_encode_1, long_sess_pos, needpos=False) # [batchsize, sess_num*sess_len, embed_dim]
        long_encode_2 = long_encode_2.view(-1, self.max_sess_num*self.max_sess_len, self.embed_dim)
        # long_encode_2, *_ = self.encoder_session(long_encode_1, long_pos, needpos=False) # [batchsize, sess_num*sess_len, embed_dim]

        
        # history-level transformer
        long_plus_query = torch.cat([long_encode_2, query_short_encode_2], 1)  # [batch, sess_num*sesslen+1, embed_dim]
        long_plus_pos = torch.cat([long_pos, torch.ones((query.size()[0], 1), dtype=torch.long).cuda()*(self.max_sess_num*self.max_sess_len+1)], 1) # query.size()[0] = batchsize
        long_encode_3, *_ = self.encoder_history(long_plus_query, long_plus_pos, needpos=False) # 这里的pos再考虑一下怎么设置？
        long_encode_3, query_long_encode_3 = torch.split(long_encode_3, [self.max_sess_num*self.max_sess_len, 1], 1) # [batchsize, 1, embed_dim]



        long_encode_2 = long_encode_2.repeat(cand_num, 1, 1) # [batchsize*(npratio+1), sess_num*sesslen, embed_dim]
        long_plus_pos = long_plus_pos.repeat(cand_num, 1)
        long_plus_doc = torch.cat([long_encode_2, doc_encode_1_sum.unsqueeze(1)], 1)
        long_doc_encode_3, *_ = self.encoder_history(long_plus_doc, long_plus_pos, needpos=False)
        _, doc_long_encode_3 = torch.split(long_doc_encode_3, [self.max_sess_num*self.max_sess_len, 1], 1)
        
        doc_encode_1_sum = doc_encode_1_sum.view(-1, cand_num, self.embed_dim)  # [batchsize, npratio+1, embed_dim]
        doc_long_encode_3 = doc_long_encode_3.view(-1, cand_num, self.embed_dim)



        # compute matching scores between short/long-term intent and document
        score_q_doc = torch.cosine_similarity(query_short_encode_2, doc_encode_1_sum, dim=2).unsqueeze(2) # [batchsize, npratio+1, 1]
        score_q_doc_his = torch.cosine_similarity(query_short_encode_2, doc_long_encode_3, dim=2).unsqueeze(2)

        score_q_his_doc = torch.cosine_similarity(query_long_encode_3, doc_encode_1_sum, dim=2).unsqueeze(2) # [batchsize, npratio+1, 1]
        score_q_his_doc_his = torch.cosine_similarity(query_long_encode_3, doc_long_encode_3, dim=2).unsqueeze(2) # [batchsize, npratio+1, 1]


        joint_score = self.joint_score_layer(torch.cat([score_q_doc, score_q_doc_his, score_q_his_doc, score_q_his_doc_his, feature], 2)).squeeze(-1) # [batchsize, nprario+1]
        pred_score = torch.sigmoid(joint_score)
        preds = F.softmax(joint_score, 1)
        joint_loss = self.criterion(joint_score, label)

        return pred_score, pred_score, preds, joint_loss, joint_loss, joint_loss