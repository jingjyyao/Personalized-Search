# 实现K-NRM模型，端到端地训练personal word embedding，并且实现global和personal 2*2的交互。
import time
import pickle
import tensorflow as tf
import numpy as np
import os, sys
import argparse
import logging
from metrics import *
from dataset import Dataset
from transformer import *
from Seq_decoder import *
from tensorflow.python.ops.rnn_cell import MultiRNNCell

class KNRM:
    def __init__(self, args):
        # logging
        self.logger = logging.getLogger('knrm')

        # basic configuration
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.n_bins = args.n_bins
        self.max_q_len = args.max_q_len
        self.max_d_len = args.max_d_len
        self.embedding_size = args.embedding_size
        self.vocabulary_size = args.vocabulary_size
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.embedding_path = args.embedding_path
        self.vocab_path = args.vocab_path
        self.feature_size = args.feature_size
        self.doc_atten = args.doc_attention
        self.use_transformer = args.use_transformer

        # settings for query suggestion decoder
        # 在vocabulary的末尾加入两个标记字符，标记decoder的开始和结束
        self.start_token = self.vocabulary_size
        self.end_token = self.vocabulary_size + 1
        self.vocabulary_size += 2
        self.hidden_size = args.hidden_size


        # get the mu and sigmas for each guassian kernel
        self.mus = self.kernel_mus(self.n_bins, use_exact=True)
        self.sigmas = self.kernel_sigmas(self.n_bins, lamb=0.5, use_exact=True)

        # build the model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.build_model()
        self.sess.run(tf.global_variables_initializer())

        # save info
        self.model_dir = args.model_dir
        self.saver = tf.train.Saver()        

    def kernel_mus(self, n_kernels, use_exact=True):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        param n_kernels: number of kernels (including exact match).
        return: l_mus, a list of mu
        """
        if use_exact:
            l_mus = [1]
        else:
            l_mus = [2]
        if n_kernels == 1:
            return l_mus
        bin_size = 2.0 / (n_kernels - 1)  # mu的间隔大小
        l_mus.append(1 - bin_size / 2) # mu is the middle of the bin
        for i in range(1, n_kernels - 1):
            l_mus.append(l_mus[i] - bin_size)
        return l_mus

    def kernel_sigmas(self, n_kernels, lamb, use_exact):
        """
        get sigmas for each guassian kernel.
        return: l_sigmas, a list of sigma
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigmas = [0.001] # for exact match. small variance
        if n_kernels == 1:
            return l_sigmas
        l_sigmas += [0.1] * (n_kernels - 1) #[bin_size * lamb] * (n_kernels - 1)
        return l_sigmas

    def gen_mask(self, Q, D, use_exact=True):
        """
        generate mask for the batch, mask padding words.
        param Q: a batch of queries, [batch_size, max_q_len]
        param D: a batch of documents, [batch_size, max_d_len]
        return: a mask of shape [batch_size, max_q_len, max_d_len].
        """
        mask = np.zeros((self.batch_size, self.max_q_len, self.max_d_len))
        Q = np.array(Q)
        D = np.array(D)
        for b in range(len(Q)):  # iter a batch
            for q in range(len(Q[b])):
                if Q[b, q] > 0:  # 此处query词有效
                    mask[b, q, D[b] > 0] = 1
                    if not use_exact:
                        mask[b, q, D[b] == Q[b, q]] = 0
        return mask

    def gen_doc_mask(self, D):
        """
        generate mask for the batch document
        return: a mask of shape [batch_size, max_d_len, 1]
        """
        mask = np.zeros((self.batch_size, self.max_d_len))
        D = np.array(D)
        for b in range(len(D)):
            mask[b, D[b]>0] = 1
        return np.reshape(mask, (self.batch_size, self.max_d_len, 1))

    def load_embedding(self, embedding_path, vocab_path): # load the pretrained embedding
        vocabulary = pickle.load(open(vocab_path, 'rb'))
        print("Successfully load the vocabulary, #words {}".format(len(vocabulary)))
        #embed = pickle.load(open(embedding_path, 'rb'))
        embed = np.random.uniform(low=-1, high=1, size=(self.vocabulary_size, self.embedding_size))
        embed[0] = np.zeros((1, self.embedding_size))  # pad的embedding为0
        with open(embedding_path, 'r') as fr:
            for line in fr:
                line = line.strip().split()
                wordid = vocabulary[line[0]]
                wordvec = np.array([float(t) for t in line[1:]])
                embed[wordid, :] = wordvec
        print("Successfully load the word vectors...")
        return embed

    def pairwise_loss(self, score1, score2):
        return (1/(1+tf.exp(score2 -score1)))

    def gen_atten_mask(self, D, max_len):
        """
        generate mask for the attention, mask padding words.
        return: an attention mask of shape [batchsize, max_d_len, max_d_len]
        """
        mask = np.zeros((self.batch_size, max_len, max_len))
        D = np.array(D)
        for b in range(len(D)):
            for d in range(len(D[b])):
                if D[b, d] > 0:
                    mask[b, d, D[b]>0] = 1
        return mask

    def self_attention(self, K, atten_mask): # 在输入的K(可谓query或document的表示)上进行self-attention处理，得到进一步的语义表示
        attention = tf.nn.softmax(tf.matmul(K, tf.transpose(K, perm=[0, 2, 1])), axis=2)
        output = tf.matmul(atten_mask * attention, K)
        return output

    def build_model(self):  # 搭建整个模型
        # input placeholder
        self.input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')
        self.input_sigma = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_sigma')

        self.input_q = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len], name='input_q')
        self.input_q_personal = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len], name='input_q_personal')
        self.input_q_weight = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len], name='q_idf')

        self.input_pos_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len], name='input_pos_d')
        self.input_neg_d = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len], name='input_neg_d')
        self.input_pos_d_personal = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len], name='input_pos_d_personal')  # 包含个性化词的id
        self.input_neg_d_personal = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_d_len], name='input_neg_d_personal')

        self.input_pos_f = tf.placeholder(tf.float32, shape=[self.batch_size, self.feature_size], name='input_pos_f')
        self.input_neg_f = tf.placeholder(tf.float32, shape=[self.batch_size, self.feature_size], name='input_neg_f')

        self.input_Y = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_Y')
        self.lambdas = tf.placeholder(tf.float32, shape=[self.batch_size], name='lambdas')

        # mask similarity of padding terms
        self.input_mask_pos = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len], name='input_mask_pos')
        self.input_mask_neg = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_d_len], name='input_mask_neg')

        # mask self attention of padding items
        self.atten_mask_q = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_q_len], name='atten_mask_q')
        self.atten_mask_pos = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len, self.max_d_len], name='atten_mask_pos')
        self.atten_mask_neg = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len, self.max_d_len], name='atten_mask_neg')
        self.atten_mask_q_personal = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_q_len, self.max_q_len], name='atten_mask_q_personal')
        self.atten_mask_pos_personal = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len, self.max_d_len], name='atten_mask_pos_personal')
        self.atten_mask_neg_personal = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_d_len, self.max_d_len], name='atten_mask_neg_personal')

        # mask the document embedding
        self.pos_d_mask = tf.placeholder(tf.float32, shape = [self.batch_size, self.max_d_len, 1], name='pos_d_mask')
        self.neg_d_mask = tf.placeholder(tf.float32, shape = [self.batch_size, self.max_d_len, 1], name='neg_d_mask')


        # initialize the embedding
        if self.embedding_path:
            embed = self.load_embedding(self.embedding_path, self.vocab_path)
            self.embeddings = tf.Variable(tf.constant(embed, dtype='float32', shape=[self.vocabulary_size, self.embedding_size]), trainable=True)
            print("Initialized embeddings with pretrained in {0}".format(self.embedding_path))
        else:
            self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size]), trainable=True)

        #look up embeddings for each term. [batchsize, qlen/dlen, emb_dim]
        self.q_embed = tf.nn.embedding_lookup(self.embeddings, self.input_q, name='q_emb')
        self.pos_d_embed = tf.nn.embedding_lookup(self.embeddings, self.input_pos_d, name='pos_d_emb')
        self.neg_d_embed = tf.nn.embedding_lookup(self.embeddings, self.input_neg_d, name='neg_d_emb')
        print("shape of q_embed: ", self.q_embed.shape)  #[batchsize, qlen, emb_dim]
        print("shape of d_embed: ", self.pos_d_embed.shape)  #[batchsize, dlen, emb_dim]

        self.q_embed_personal = tf.nn.embedding_lookup(self.embeddings, self.input_q_personal, name='q_emb_personal')
        self.pos_d_embed_personal = tf.nn.embedding_lookup(self.embeddings, self.input_pos_d_personal, name='pos_d_emb_personal')
        self.neg_d_embed_personal = tf.nn.embedding_lookup(self.embeddings, self.input_neg_d_personal, name='neg_d_emb_personal')
        print("shape of q_embed_personal: ", self.q_embed_personal.shape)  #[batchsize, qlen, emb_dim]
        print("shape of d_embed_personal: ", self.pos_d_embed_personal.shape)  #[batchsize, dlen, emb_dim]        


        # common tensors
        mu = tf.reshape(self.input_mu, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(self.input_sigma, shape=[1, 1, self.n_bins])
        rs_pos_mask = tf.reshape(self.input_mask_pos, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_neg_mask = tf.reshape(self.input_mask_neg, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])
        rs_q_weight = tf.reshape(self.input_q_weight, shape=[self.batch_size, self.max_q_len, 1])


        # compute the similarity matrix
        # Interaction 1： global with global
        with tf.variable_scope("gloabl_global"):
            norm_q = tf.sqrt(tf.reduce_sum(tf.square(self.q_embed), 2, keep_dims=True))
            normalized_q_embed = self.q_embed / (norm_q + 1e-5)  # 为了避免出现nan

            norm_pos_d = tf.sqrt(tf.reduce_sum(tf.square(self.pos_d_embed), 2, keep_dims=True))
            normalized_pos_d_embed = self.pos_d_embed / (norm_pos_d + 1e-5)
            tmp_pos_d = tf.transpose(normalized_pos_d_embed, perm=[0, 2, 1])

            norm_neg_d = tf.sqrt(tf.reduce_sum(tf.square(self.neg_d_embed), 2, keep_dims=True))
            normalized_neg_d_embed = self.neg_d_embed / (norm_neg_d + 1e-5)
            tmp_neg_d = tf.transpose(normalized_neg_d_embed, perm=[0, 2, 1])
            print("shape of tmp_pos_d: ", tmp_pos_d.shape) #[batchsize, embed_dim, dlen]

            # similarity matrix [batchsize, qlen, dlen]
            self.pos_similarity_1 = tf.matmul(normalized_q_embed, tmp_pos_d, name='pos_similarity_matrix_1')
            self.neg_similarity_1 = tf.matmul(normalized_q_embed, tmp_neg_d, name='neg_similarity_matrix_1')
            print("shape of similarity matrix: ", self.pos_similarity_1.shape)  #[batchsize, qlen, dlen]

            # compute the Gaussian scores of each kernel [batchsize, qlen, dlen, n_bins]                
            rs_pos_sim = tf.reshape(self.pos_similarity_1, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])
            rs_neg_sim = tf.reshape(self.neg_similarity_1, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])

            self.pos_kernel_1 = tf.exp(-tf.square(rs_pos_sim - mu) / (tf.square(sigma) * 2))
            self.neg_kernel_1 = tf.exp(-tf.square(rs_neg_sim - mu) / (tf.square(sigma) * 2))
            print("shape of doc kernels: ", self.pos_kernel_1.shape)  #[batchsize, qlen, dlen, n_bins]

            # mask the non-existing words
            masked_pos_kernel = rs_pos_mask * self.pos_kernel_1
            masked_neg_kernel = rs_neg_mask * self.neg_kernel_1

            # sum up the Gaussian scores [batchsize, qlen, n_bins]
            sum_pos_kernel = tf.reduce_sum(masked_pos_kernel, 2)
            sum_pos_kernel = tf.log(tf.maximum(sum_pos_kernel, 1e-10)) * 0.01 # 0.01 used to scale the data
            sum_neg_kernel = tf.reduce_sum(masked_neg_kernel, 2)
            sum_neg_kernel = tf.log(tf.maximum(sum_neg_kernel, 1e-10)) * 0.01
            print("shape of doc sum_kernels: ", sum_pos_kernel.shape)  #[batchsize, qlen, n_bins]


            self.aggregate_pos_kernel_1 = tf.reduce_sum(rs_q_weight * sum_pos_kernel, 1)
            self.aggregate_neg_kernel_1 = tf.reduce_sum(rs_q_weight * sum_neg_kernel, 1)
            print("shape of aggregate kernels: ", self.aggregate_pos_kernel_1.shape)  #[batchsize, n_bins]

            with tf.variable_scope("Interact_Score_1"):
                self.pos_kernel_score_1 = tf.layers.dense(inputs=self.aggregate_pos_kernel_1, units=1, activation=tf.nn.tanh, name="interact_score_1")
            with tf.variable_scope("Interact_Score_1", reuse=True):
                self.neg_kernel_score_1 = tf.layers.dense(inputs=self.aggregate_neg_kernel_1, units=1, activation=tf.nn.tanh, name="interact_score_1")



        # Interaction 2： personal with personal
        with tf.variable_scope("personal_personal"):
            norm_q = tf.sqrt(tf.reduce_sum(tf.square(self.q_embed_personal), 2, keep_dims=True))
            normalized_q_embed = self.q_embed_personal / (norm_q + 1e-5)  # 为了避免出现nan

            norm_pos_d = tf.sqrt(tf.reduce_sum(tf.square(self.pos_d_embed_personal), 2, keep_dims=True))
            normalized_pos_d_embed = self.pos_d_embed_personal / (norm_pos_d + 1e-5)
            tmp_pos_d = tf.transpose(normalized_pos_d_embed, perm=[0, 2, 1])

            norm_neg_d = tf.sqrt(tf.reduce_sum(tf.square(self.neg_d_embed_personal), 2, keep_dims=True))
            normalized_neg_d_embed = self.neg_d_embed_personal / (norm_neg_d + 1e-5)
            tmp_neg_d = tf.transpose(normalized_neg_d_embed, perm=[0, 2, 1])
            print("shape of tmp_pos_d: ", tmp_pos_d.shape) #[batchsize, embed_dim, dlen]

            # similarity matrix [batchsize, qlen, dlen]
            self.pos_similarity_2 = tf.matmul(normalized_q_embed, tmp_pos_d, name='pos_similarity_matrix_1')
            self.neg_similarity_2 = tf.matmul(normalized_q_embed, tmp_neg_d, name='neg_similarity_matrix_1')
            print("shape of similarity matrix: ", self.pos_similarity_2.shape)  #[batchsize, qlen, dlen]

            # compute the Gaussian scores of each kernel [batchsize, qlen, dlen, n_bins]                
            rs_pos_sim = tf.reshape(self.pos_similarity_2, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])
            rs_neg_sim = tf.reshape(self.neg_similarity_2, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])

            self.pos_kernel_2 = tf.exp(-tf.square(rs_pos_sim - mu) / (tf.square(sigma) * 2))
            self.neg_kernel_2 = tf.exp(-tf.square(rs_neg_sim - mu) / (tf.square(sigma) * 2))
            print("shape of doc kernels: ", self.pos_kernel_2.shape)  #[batchsize, qlen, dlen, n_bins]

            # mask the non-existing words
            masked_pos_kernel = rs_pos_mask * self.pos_kernel_2
            masked_neg_kernel = rs_neg_mask * self.neg_kernel_2

            # sum up the Gaussian scores [batchsize, qlen, n_bins]
            sum_pos_kernel = tf.reduce_sum(masked_pos_kernel, 2)
            sum_pos_kernel = tf.log(tf.maximum(sum_pos_kernel, 1e-10)) * 0.01 # 0.01 used to scale the data
            sum_neg_kernel = tf.reduce_sum(masked_neg_kernel, 2)
            sum_neg_kernel = tf.log(tf.maximum(sum_neg_kernel, 1e-10)) * 0.01
            print("shape of doc sum_kernels: ", sum_pos_kernel.shape)  #[batchsize, qlen, n_bins]


            self.aggregate_pos_kernel_2 = tf.reduce_sum(rs_q_weight * sum_pos_kernel, 1)
            self.aggregate_neg_kernel_2 = tf.reduce_sum(rs_q_weight * sum_neg_kernel, 1)
            print("shape of aggregate kernels: ", self.aggregate_pos_kernel_2.shape)  #[batchsize, n_bins]

            with tf.variable_scope("Interact_Score_2"):
                self.pos_kernel_score_2 = tf.layers.dense(inputs=self.aggregate_pos_kernel_2, units=1, activation=tf.nn.tanh, name="interact_score_2")
            with tf.variable_scope("Interact_Score_2", reuse=True):
                self.neg_kernel_score_2 = tf.layers.dense(inputs=self.aggregate_neg_kernel_2, units=1, activation=tf.nn.tanh, name="interact_score_2")



        # Interaction 3: attend global with attend global
        # a self-attention layer on the document
        with tf.variable_scope("attend_global_global"):
            #with tf.variable_scope("query_attention"):
            with tf.variable_scope("global_attention"):
                if not self.use_transformer:
                    self.q_embed_attend = attention_layer(self.q_embed, self.q_embed, self.atten_mask_q, num_attention_heads=8, size_per_head=50, 
                                                          batch_size=self.batch_size, from_seq_length=self.max_q_len, to_seq_length=self.max_q_len)
                else:
                    self.q_embed_attend = transformer_model(self.q_embed, self.atten_mask_q, hidden_size=self.embedding_size, attention_hidden_size=400, 
                                                          num_hidden_layers=2, num_attention_heads=8, intermediate_size=400,)
            #with tf.variable_scope("doc_attention"):
            with tf.variable_scope("global_attention", reuse=True):
                if not self.use_transformer:
                    self.pos_d_embed_attend = attention_layer(self.pos_d_embed, self.pos_d_embed, self.atten_mask_pos, num_attention_heads=8, size_per_head=50, 
                                                              batch_size=self.batch_size, from_seq_length=self.max_d_len, to_seq_length=self.max_d_len)
                else:
                    self.pos_d_embed_attend = transformer_model(self.pos_d_embed, self.atten_mask_pos, hidden_size=self.embedding_size, attention_hidden_size=400, 
                                                          num_hidden_layers=2, num_attention_heads=8, intermediate_size=400,)

            #with tf.variable_scope("doc_attention", reuse=True):
            with tf.variable_scope("global_attention", reuse=True):
                if not self.use_transformer:
                    self.neg_d_embed_attend = attention_layer(self.neg_d_embed, self.neg_d_embed, self.atten_mask_neg, num_attention_heads=8, size_per_head=50, 
                                                              batch_size=self.batch_size, from_seq_length=self.max_d_len, to_seq_length=self.max_d_len)
                else:
                    self.neg_d_embed_attend = transformer_model(self.neg_d_embed, self.atten_mask_neg, hidden_size=self.embedding_size, attention_hidden_size=400, 
                                                          num_hidden_layers=2, num_attention_heads=8, intermediate_size=400,)
            print("shape of attended embed: ", self.pos_d_embed_attend.shape)

            with tf.variable_scope("attend_knrm"):
                norm_q = tf.sqrt(tf.reduce_sum(tf.square(self.q_embed_attend), 2, keep_dims=True))
                normalized_q_embed = self.q_embed_attend / (norm_q + 1e-5)  # 为了避免出现nan

                norm_pos_d = tf.sqrt(tf.reduce_sum(tf.square(self.pos_d_embed_attend), 2, keep_dims=True))
                normalized_pos_d_embed = self.pos_d_embed_attend / (norm_pos_d + 1e-5)
                tmp_pos_d = tf.transpose(normalized_pos_d_embed, perm=[0, 2, 1])

                norm_neg_d = tf.sqrt(tf.reduce_sum(tf.square(self.neg_d_embed_attend), 2, keep_dims=True))
                normalized_neg_d_embed = self.neg_d_embed_attend / (norm_neg_d + 1e-5)
                tmp_neg_d = tf.transpose(normalized_neg_d_embed, perm=[0, 2, 1])
                print("shape of tmp_pos_d_attend: ", tmp_pos_d.shape) #[batchsize, embed_dim, dlen]

                # similarity matrix [batchsize, qlen, dlen]
                self.pos_similarity_attend = tf.matmul(normalized_q_embed, tmp_pos_d, name='pos_similarity_matrix_attend')
                self.neg_similarity_attend = tf.matmul(normalized_q_embed, tmp_neg_d, name='neg_similarity_matrix_attend')
                print("shape of attend similarity matrix: ", self.pos_similarity_attend.shape)  #[batchsize, qlen, dlen]

                rs_pos_sim = tf.reshape(self.pos_similarity_attend, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])
                rs_neg_sim = tf.reshape(self.neg_similarity_attend, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])

                self.pos_kernel_attend = tf.exp(-tf.square(rs_pos_sim - mu) / (tf.square(sigma) * 2))
                self.neg_kernel_attend = tf.exp(-tf.square(rs_neg_sim - mu) / (tf.square(sigma) * 2))
                print("shape of attend doc kernels: ", self.pos_kernel_attend.shape)  #[batchsize, qlen, dlen, n_bins]

                # mask the non-existing words
                masked_pos_kernel = rs_pos_mask * self.pos_kernel_attend
                masked_neg_kernel = rs_neg_mask * self.neg_kernel_attend

                # sum up the Gaussian scores [batchsize, qlen, n_bins]
                sum_pos_kernel = tf.reduce_sum(masked_pos_kernel, 2)
                sum_pos_kernel = tf.log(tf.maximum(sum_pos_kernel, 1e-10)) * 0.01 # 0.01 used to scale the data
                sum_neg_kernel = tf.reduce_sum(masked_neg_kernel, 2)
                sum_neg_kernel = tf.log(tf.maximum(sum_neg_kernel, 1e-10)) * 0.01
                print("shape of attend doc sum_kernels: ", sum_pos_kernel.shape)  #[batchsize, qlen, n_bins]

                self.aggregate_pos_kernel_attend = tf.reduce_sum(rs_q_weight * sum_pos_kernel, 1)
                self.aggregate_neg_kernel_attend = tf.reduce_sum(rs_q_weight * sum_neg_kernel, 1)
                print("shape of attend aggregate kernels: ", self.aggregate_pos_kernel_attend.shape)  #[batchsize, n_bins]


                with tf.variable_scope("Interact_Score_Attend"):
                    self.pos_kernel_score_attend = tf.layers.dense(inputs=self.aggregate_pos_kernel_attend, units=1, activation=tf.nn.tanh, name="interact_score_attend")
                with tf.variable_scope("Interact_Score_Attend", reuse=True):
                    self.neg_kernel_score_attend = tf.layers.dense(inputs=self.aggregate_neg_kernel_attend, units=1, activation=tf.nn.tanh, name="interact_score_attend")



        # Interaction 4: attend personal with attend personal
        # a self-attention layer on the document
        with tf.variable_scope("attend_personal_personal"):
            #with tf.variable_scope("query_attention"):
            with tf.variable_scope("personal_attention"):
                if not self.use_transformer:
                    self.q_embed_attend_personal = attention_layer(self.q_embed_personal, self.q_embed_personal, self.atten_mask_q_personal, num_attention_heads=8, size_per_head=50, 
                                                        batch_size=self.batch_size, from_seq_length=self.max_q_len, to_seq_length=self.max_q_len)
                else:
                    self.q_embed_attend_personal = transformer_model(self.q_embed_personal, self.atten_mask_q_personal, hidden_size=self.embedding_size, attention_hidden_size=400, 
                                                          num_hidden_layers=1, num_attention_heads=8, intermediate_size=400,)
            with tf.variable_scope("personal_attention", reuse=True):
            #with tf.variable_scope("doc_attention"):
                if not self.use_transformer:
                    self.pos_d_embed_attend_personal = attention_layer(self.pos_d_embed_personal, self.pos_d_embed_personal, self.atten_mask_pos_personal, num_attention_heads=8, size_per_head=50, 
                                                          batch_size=self.batch_size, from_seq_length=self.max_d_len, to_seq_length=self.max_d_len)
                else:
                    self.pos_d_embed_attend_personal = transformer_model(self.pos_d_embed_personal, self.atten_mask_pos_personal, hidden_size=self.embedding_size, attention_hidden_size=400, 
                                                          num_hidden_layers=1, num_attention_heads=8, intermediate_size=400,)

            with tf.variable_scope("personal_attention", reuse=True):
            #with tf.variable_scope("doc_attention", reuse=True):
                if not self.use_transformer:
                    self.neg_d_embed_attend_personal = attention_layer(self.neg_d_embed_personal, self.neg_d_embed_personal, self.atten_mask_neg_personal, num_attention_heads=8, size_per_head=50, 
                                                          batch_size=self.batch_size, from_seq_length=self.max_d_len, to_seq_length=self.max_d_len)
                else:
                    self.neg_d_embed_attend_personal = transformer_model(self.neg_d_embed_personal, self.atten_mask_neg_personal, hidden_size=self.embedding_size, attention_hidden_size=400, 
                                                          num_hidden_layers=1, num_attention_heads=8, intermediate_size=400,)
            print("shape of attended embed: ", self.pos_d_embed_attend.shape)

            with tf.variable_scope("attend_knrm"):
                norm_q = tf.sqrt(tf.reduce_sum(tf.square(self.q_embed_attend_personal), 2, keep_dims=True))
                normalized_q_embed = self.q_embed_attend_personal / (norm_q + 1e-5)  # 为了避免出现nan

                norm_pos_d = tf.sqrt(tf.reduce_sum(tf.square(self.pos_d_embed_attend_personal), 2, keep_dims=True))
                normalized_pos_d_embed = self.pos_d_embed_attend_personal / (norm_pos_d + 1e-5)
                tmp_pos_d = tf.transpose(normalized_pos_d_embed, perm=[0, 2, 1])

                norm_neg_d = tf.sqrt(tf.reduce_sum(tf.square(self.neg_d_embed_attend_personal), 2, keep_dims=True))
                normalized_neg_d_embed = self.neg_d_embed_attend_personal / (norm_neg_d + 1e-5)
                tmp_neg_d = tf.transpose(normalized_neg_d_embed, perm=[0, 2, 1])
                print("shape of tmp_pos_d_attend: ", tmp_pos_d.shape) #[batchsize, embed_dim, dlen]

                # similarity matrix [batchsize, qlen, dlen]
                self.pos_similarity_attend_personal = tf.matmul(normalized_q_embed, tmp_pos_d, name='pos_similarity_matrix_attend_personal')
                self.neg_similarity_attend_personal = tf.matmul(normalized_q_embed, tmp_neg_d, name='neg_similarity_matrix_attend_personal')
                print("shape of attend similarity matrix: ", self.pos_similarity_attend.shape)  #[batchsize, qlen, dlen]

                rs_pos_sim = tf.reshape(self.pos_similarity_attend_personal, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])
                rs_neg_sim = tf.reshape(self.neg_similarity_attend_personal, shape=[self.batch_size, self.max_q_len, self.max_d_len, 1])

                self.pos_kernel_attend_personal = tf.exp(-tf.square(rs_pos_sim - mu) / (tf.square(sigma) * 2))
                self.neg_kernel_attend_personal = tf.exp(-tf.square(rs_neg_sim - mu) / (tf.square(sigma) * 2))
                print("shape of attend doc kernels: ", self.pos_kernel_attend_personal.shape)  #[batchsize, qlen, dlen, n_bins]

                # mask the non-existing words
                masked_pos_kernel = rs_pos_mask * self.pos_kernel_attend_personal
                masked_neg_kernel = rs_neg_mask * self.neg_kernel_attend_personal

                # sum up the Gaussian scores [batchsize, qlen, n_bins]
                sum_pos_kernel = tf.reduce_sum(masked_pos_kernel, 2)
                sum_pos_kernel = tf.log(tf.maximum(sum_pos_kernel, 1e-10)) * 0.01 # 0.01 used to scale the data
                sum_neg_kernel = tf.reduce_sum(masked_neg_kernel, 2)
                sum_neg_kernel = tf.log(tf.maximum(sum_neg_kernel, 1e-10)) * 0.01
                print("shape of attend doc sum_kernels: ", sum_pos_kernel.shape)  #[batchsize, qlen, n_bins]

                self.aggregate_pos_kernel_attend_personal = tf.reduce_sum(rs_q_weight * sum_pos_kernel, 1)
                self.aggregate_neg_kernel_attend_personal = tf.reduce_sum(rs_q_weight * sum_neg_kernel, 1)
                print("shape of attend aggregate kernels: ", self.aggregate_pos_kernel_attend_personal.shape)  #[batchsize, n_bins]


                with tf.variable_scope("Interact_Score_Attend_Personal"):
                    self.pos_kernel_score_attend_personal = tf.layers.dense(inputs=self.aggregate_pos_kernel_attend_personal, units=1, activation=tf.nn.tanh, name="interact_score_attend_personal")
                with tf.variable_scope("Interact_Score_Attend_Personal", reuse=True):
                    self.neg_kernel_score_attend_personal = tf.layers.dense(inputs=self.aggregate_neg_kernel_attend_personal, units=1, activation=tf.nn.tanh, name="interact_score_attend_personal")



        # Multitask, 增加query suggestion的部分
        with tf.variable_scope("query_suggestion"):
            self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="encoder_inputs_length")
            #self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_q_len], name="decoder_inputs")
            self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name="decoder_inputs") # max length是动态变化的
            self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="decoder_inputs_length")
            decoder_start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * self.start_token
            decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * self.end_token

            # insert start symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token, self.decoder_inputs], axis=1)
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1
            self.decoder_targets_train = tf.concat([self.decoder_inputs, decoder_end_token], axis=1)

            #input_layer = Dense(self.hidden_size, dtype=tf.float32, name="input_layer")
            #self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs_train, name='decoder_inputs_embedded')
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs, name='decoder_inputs_embedded')
            print("shape of decoder inputs embeded: ", self.decoder_inputs_embedded.shape)
            #self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

            # encode the query word vectors to hidden state
            init_state = tf.get_variable('encode_init', [self.batch_size, self.hidden_size], dtype=np.float32)
            #self.encoder_cell = MultiRNNCell([tf.contrib.rnn.GRUCell(num_units=self.hidden_size)])
            cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            self.encoder_outputs, self.hidden_state = tf.nn.dynamic_rnn(cell=cell,
                                                                        #inputs=self.q_embed,
                                                                        inputs=self.q_embed_personal,
                                                                        sequence_length=self.encoder_inputs_length,
                                                                        initial_state=init_state,
                                                                        scope="encoder")
            print("shape of encoder outputs: ", self.encoder_outputs.shape)
            print("shape of hidden state: ", self.hidden_state.shape)
            # encode_output: [batch_size, max_q_len, hidden_size], hidden_state: [batch_size, hidden_size]

            # decode the hidden state to a sequence
            #decoder_init_state = tf.get_variable('decode_init', [self.batch_size, self.hidden_size], dtype=np.float32)
            #decoder_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            self.decoder_outputs, self.decoder_hidden_state = tf.nn.dynamic_rnn(cell=cell,
                                                                        inputs=self.decoder_inputs_embedded,
                                                                        #inputs=self.q_embed_personal,
                                                                        sequence_length=self.decoder_inputs_length,
                                                                        initial_state=init_state,
                                                                        scope="decoder")
            print("shape of decoder hidden state: ", self.decoder_hidden_state.shape)

            self.qs_loss = - tf.reduce_mean(self.hidden_state * self.decoder_hidden_state)
            # self.qs_loss = decode_sequence(hidden_units=self.hidden_size, num_decoder_symbols=self.vocabulary_size, 
            #                decoder_inputs_embedded=self.decoder_inputs_embedded, decoder_inputs_length_train=self.decoder_inputs_length_train,
            #                decoder_targets_train=self.decoder_targets_train, encoder_outputs=self.encoder_outputs, encoder_last_state=self.hidden_state, 
            #                encoder_inputs_length=self.encoder_inputs_length, batch_size=self.batch_size)

            with tf.variable_scope("suggest_query_match"):
                self.pos_d_vector = tf.reduce_sum(self.pos_d_embed_personal * self.pos_d_mask, axis=1) # [batch_size, embedding_size]
                self.neg_d_vector = tf.reduce_sum(self.neg_d_embed_personal * self.neg_d_mask, axis=1) # [batch_size, embedding_size]
                # self.pos_d_vector = tf.reduce_sum(self.pos_d_embed * self.pos_d_mask, axis=1) # [batch_size, embedding_size]
                # self.neg_d_vector = tf.reduce_sum(self.neg_d_embed * self.neg_d_mask, axis=1) # [batch_size, embedding_size]
                with tf.variable_scope("document_encode"):
                    self.pos_d_vector = tf.layers.dense(inputs=self.pos_d_vector, units=self.hidden_size, activation=tf.nn.tanh, name="document_encodes")
                with tf.variable_scope("document_encode", reuse=True):
                    self.neg_d_vector = tf.layers.dense(inputs=self.neg_d_vector, units=self.hidden_size, activation=tf.nn.tanh, name="document_encodes")
                norm_suggest_q = tf.sqrt(tf.reduce_sum(tf.square(self.hidden_state), 1, keep_dims=True)) # [batch_size, ]
                normalized_suggest_q = self.hidden_state / (norm_suggest_q + 1e-5)

                norm_pos_d_vector = tf.sqrt(tf.reduce_sum(tf.square(self.pos_d_vector), 1, keep_dims=True)) # [batch_size, ]
                normalized_pos_d_vector = self.pos_d_vector / (norm_pos_d_vector + 1e-5)  

                norm_neg_d_vector = tf.sqrt(tf.reduce_sum(tf.square(self.neg_d_vector), 1, keep_dims=True)) # [batch_size, ]
                normalized_neg_d_vector = self.neg_d_vector / (norm_neg_d_vector + 1e-5)

                self.pos_suggest_score = tf.reduce_sum(normalized_suggest_q * normalized_pos_d_vector, axis=1, keep_dims=True) 
                self.neg_suggest_score = tf.reduce_sum(normalized_suggest_q * normalized_neg_d_vector, axis=1, keep_dims=True) 



        with tf.variable_scope("Feature_Score"):
            self.pos_feature_score = tf.layers.dense(inputs=self.input_pos_f, units=1, activation=tf.nn.tanh, name="feature_score")
        with tf.variable_scope("Feature_Score", reuse=True):
            self.neg_feature_score = tf.layers.dense(inputs=self.input_neg_f, units=1, activation=tf.nn.tanh, name="feature_score")

        with tf.variable_scope("Final_Score"):
            #final_pos_score = tf.concat([self.pos_kernel_score_1, self.pos_kernel_score_attend, self.pos_suggest_score, self.pos_feature_score], -1)
            final_pos_score = tf.concat([self.pos_kernel_score_1, self.pos_kernel_score_2, self.pos_kernel_score_attend, self.pos_kernel_score_attend_personal, self.pos_suggest_score, self.pos_feature_score], -1)
            self.pos_score = tf.layers.dense(inputs=final_pos_score, units=1, activation=None, name="final_score")
        with tf.variable_scope("Final_Score", reuse=True):
            #final_neg_score = tf.concat([self.neg_kernel_score_1, self.neg_kernel_score_attend, self.neg_suggest_score, self.neg_feature_score], -1)
            final_neg_score = tf.concat([self.neg_kernel_score_1, self.neg_kernel_score_2, self.neg_kernel_score_attend, self.neg_kernel_score_attend_personal, self.neg_suggest_score, self.neg_feature_score], -1)
            self.neg_score = tf.layers.dense(inputs=final_neg_score, units=1, activation=None, name="final_score")

        # prediction and accuracy
        self.scores = tf.concat([self.pos_score, self.neg_score], 1)
        self.p_scores = tf.concat([self.pairwise_loss(self.pos_score, self.neg_score),
                        self.pairwise_loss(self.neg_score, self.pos_score)], 1)
        self.preds = tf.cast(tf.argmax(tf.nn.softmax(self.scores), 1), tf.int32)
        self.correct = tf.equal(self.preds, self.input_Y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float64))

        # loss and optimization # 问题：梯度中出现了nan，怀疑可能是因为梯度爆炸导致的，引入梯度裁剪
        #self.loss = tf.reduce_mean(tf.maximum(0.0, 1-self.pos_score + self.neg_score))
        self.loss = tf.reduce_mean(self.lambdas*tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.input_Y, logits=self.p_scores)) + self.qs_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)        
        self.gradient = self.optimizer.compute_gradients(self.loss)
        self.clipped_gradients = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in self.gradient]
        self.train_step = self.optimizer.apply_gradients(self.clipped_gradients)

    def train(self, dataset):  # 训练模型
        self.train_losses = []
        self.train_accuracies = []
        self.valid_losses = []
        self.valid_accuracies = []

        train_start_time = time.time()

        for idx, epoch_data in enumerate(dataset.gen_epochs()):
            training_loss = 0.0
            training_acc = 0.0
            training_steps = 0.0
            valid_loss = 0.0
            valid_acc = 0.0
            valid_steps = 0.0
            for batch_data in epoch_data:
                batch_start_time = time.time()
                #print("q_train: ", batch_data['q_train'][0])
                #print("d1_train: ", batch_data['d1_train'][0])
                feed_dict_train = {self.input_mu: self.mus,
                                   self.input_sigma: self.sigmas,
                                   self.input_q: batch_data['q_train'],
                                   self.input_q_personal: batch_data['q_train_personal'],
                                   self.input_q_weight: batch_data['q_weight_train'],
                                   self.input_pos_d: batch_data['d1_train'],
                                   self.input_neg_d: batch_data['d2_train'],
                                   self.input_pos_d_personal: batch_data['d1_train_personal'],
                                   self.input_neg_d_personal: batch_data['d2_train_personal'],
                                   self.input_pos_f: batch_data['f1_train'],
                                   self.input_neg_f: batch_data['f2_train'],
                                   self.encoder_inputs_length: batch_data['q_len_train'],
                                   self.decoder_inputs: batch_data['next_q_train'],
                                   self.decoder_inputs_length: batch_data['next_q_len_train'],
                                   self.input_Y: batch_data['Y_train'],
                                   self.lambdas: batch_data['lambda_train'],
                                   self.input_mask_pos: self.gen_mask(batch_data['q_train'], batch_data['d1_train']),
                                   self.input_mask_neg: self.gen_mask(batch_data['q_train'], batch_data['d2_train']),
                                   self.atten_mask_q: self.gen_atten_mask(batch_data['q_train'], self.max_q_len),
                                   self.atten_mask_pos: self.gen_atten_mask(batch_data['d1_train'], self.max_d_len),
                                   self.atten_mask_neg: self.gen_atten_mask(batch_data['d2_train'], self.max_d_len),
                                   self.atten_mask_q_personal: self.gen_atten_mask(batch_data['q_train_personal'], self.max_q_len),
                                   self.atten_mask_pos_personal: self.gen_atten_mask(batch_data['d1_train_personal'], self.max_d_len),
                                   self.atten_mask_neg_personal: self.gen_atten_mask(batch_data['d2_train_personal'], self.max_d_len),
                                   self.pos_d_mask: self.gen_doc_mask(batch_data['d1_train']),
                                   self.neg_d_mask: self.gen_doc_mask(batch_data['d2_train']),}
                qs_loss, train_loss_, train_acc_, _, q_train, d1_train, q_emb, d1_emb, pos_sim, pos_aggre, neg_aggre, scores, gradient, clipped_gradients = self.sess.run([self.qs_loss, self.loss, self.accuracy, self.train_step, self.input_q, self.input_pos_d, self.q_embed, self.pos_d_embed, self.pos_similarity_1, self.aggregate_pos_kernel_1, self.aggregate_neg_kernel_1, self.scores, self.gradient, self.clipped_gradients], feed_dict_train)
                training_loss += train_loss_
                training_acc += train_acc_
                training_steps += 1

                feed_dict_valid = {self.input_mu: self.mus,
                                   self.input_sigma: self.sigmas,
                                   self.input_q: batch_data['q_valid'],
                                   self.input_q_personal: batch_data['q_valid_personal'],
                                   self.input_q_weight: batch_data['q_weight_valid'],
                                   self.input_pos_d: batch_data['d1_valid'],
                                   self.input_neg_d: batch_data['d2_valid'],
                                   self.input_pos_d_personal: batch_data['d1_valid_personal'],
                                   self.input_neg_d_personal: batch_data['d2_valid_personal'],
                                   self.input_pos_f: batch_data['f1_valid'],
                                   self.input_neg_f: batch_data['f2_valid'],
                                   self.encoder_inputs_length: batch_data['q_len_valid'],
                                   self.decoder_inputs: batch_data['next_q_valid'],
                                   self.decoder_inputs_length: batch_data['next_q_len_valid'],
                                   self.lambdas: batch_data['lambda_valid'],
                                   self.input_Y: batch_data['Y_valid'],
                                   self.input_mask_pos: self.gen_mask(batch_data['q_valid'], batch_data['d1_valid']),
                                   self.input_mask_neg: self.gen_mask(batch_data['q_valid'], batch_data['d2_valid']),
                                   self.atten_mask_q: self.gen_atten_mask(batch_data['q_valid'], self.max_q_len),
                                   self.atten_mask_pos: self.gen_atten_mask(batch_data['d1_valid'], self.max_d_len),
                                   self.atten_mask_neg: self.gen_atten_mask(batch_data['d2_valid'], self.max_d_len),
                                   self.atten_mask_q_personal: self.gen_atten_mask(batch_data['q_valid_personal'], self.max_q_len),
                                   self.atten_mask_pos_personal: self.gen_atten_mask(batch_data['d1_valid_personal'], self.max_d_len),
                                   self.atten_mask_neg_personal: self.gen_atten_mask(batch_data['d2_valid_personal'], self.max_d_len),
                                   self.pos_d_mask: self.gen_doc_mask(batch_data['d1_valid']),
                                   self.neg_d_mask: self.gen_doc_mask(batch_data['d2_valid']),}
                valid_qs_loss, valid_loss_, valid_acc_ = self.sess.run([self.qs_loss, self.loss, self.accuracy], feed_dict_valid)
                valid_loss += valid_loss_
                valid_acc += valid_acc_
                valid_steps += 1

                if training_steps % 10 == 0:
                    #print("clipped_gradients: ", clipped_gradients)
                    print("\nQS Loss: ", qs_loss, "\tValid QS Loss: ", valid_qs_loss)
                    print("\nBatch Loss: ", train_loss_, "\tAccuracy: ", train_acc_, "\tTime cost: ", time.time()-batch_start_time, "s.")
                    print("Batch Valid Loss: ", valid_loss_, "\tAccuracy: ", valid_acc_, "\tTime cost: ", time.time()-batch_start_time, "s.")
                #exit()
                if training_steps % 100 == 0:
                    print("nan in query embedding: ", np.isnan(q_emb).any())
                    print("nan in positive document embedding: ", np.isnan(d1_emb).any())
                    print("nan in positive similarity: ", np.isnan(pos_sim).any())
                    print("nan in scores: ", np.isnan(scores).any())
                    self.logger.info('training loss before {} step: {}.'.format(training_steps, training_loss / training_steps))
                    self.logger.info('validing loss before {} step: {}.'.format(valid_steps, valid_loss / valid_steps))
            self.train_losses.append(training_loss / training_steps)
            self.train_accuracies.append(training_acc / training_steps)
            self.valid_losses.append(valid_loss / valid_steps)
            self.valid_accuracies.append(valid_acc / valid_steps)

            self.save(self.model_dir + '_' + str(idx)) 
        self.logger.info("Train losses in different epoches are " +  str(self.train_losses))
        self.logger.info("Train accuracy in different epoches are " + str(self.train_accuracies))
        self.logger.info("Valid losses in different epoches are " + str(self.valid_losses))
        self.logger.info("Valid accuracy in different epoches are " + str(self.valid_accuracies))                       

    def test(self, dataset):  # 测试模型
        testing_loss = 0.0
        testing_acc = 0.0
        testing_steps = 0
        for batch_data in dataset.gen_test_batch():
            feed_dict_test = {self.input_mu: self.mus,
                              self.input_sigma: self.sigmas,
                              self.input_q: batch_data['q_test'],
                              self.input_q_personal: batch_data['q_test_personal'],
                              self.input_q_weight: batch_data['q_weight_test'],
                              self.input_pos_d: batch_data['d1_test'],
                              self.input_neg_d: batch_data['d2_test'],
                              self.input_pos_d_personal: batch_data['d1_test_personal'],
                              self.input_neg_d_personal: batch_data['d2_test_personal'],
                              self.input_pos_f: batch_data['f1_test'],
                              self.input_neg_f: batch_data['f2_test'],
                              self.encoder_inputs_length: batch_data['q_len_test'],
                              self.decoder_inputs: batch_data['next_q_test'],
                              self.decoder_inputs_length: batch_data['next_q_len_test'],
                              self.input_Y: batch_data['Y_test'],
                              self.lambdas: batch_data['lambda_test'],
                              self.input_mask_pos: self.gen_mask(batch_data['q_test'], batch_data['d1_test']),
                              self.input_mask_neg: self.gen_mask(batch_data['q_test'], batch_data['d2_test']),
                              self.atten_mask_q: self.gen_atten_mask(batch_data['q_test'], self.max_q_len),
                              self.atten_mask_pos: self.gen_atten_mask(batch_data['d1_test'], self.max_d_len),
                              self.atten_mask_neg: self.gen_atten_mask(batch_data['d2_test'], self.max_d_len),
                              self.atten_mask_q_personal: self.gen_atten_mask(batch_data['q_test_personal'], self.max_q_len),
                              self.atten_mask_pos_personal: self.gen_atten_mask(batch_data['d1_test_personal'], self.max_d_len),
                              self.atten_mask_neg_personal: self.gen_atten_mask(batch_data['d2_test_personal'], self.max_d_len),
                              self.pos_d_mask: self.gen_doc_mask(batch_data['d1_test']),
                              self.neg_d_mask: self.gen_doc_mask(batch_data['d2_test']),}
            test_loss_, test_acc_, scores = self.sess.run([self.loss, self.accuracy, self.scores], feed_dict_test)
            #print("input_q: ", batch_data['q_test'][0], "input_d: ", batch_data['d1_test'][0])
            #print("testing scores: ", scores[0])
            testing_loss += test_loss_
            testing_acc += test_acc_
            testing_steps += 1
        self.logger.info("Average loss on test dataset: " + str(testing_loss / testing_steps))
        self.logger.info("Accuracy: " + str(testing_acc / testing_steps))

    def score(self, dataset, evaluation):  # 计算最终的排序结果
        with open('test_score.txt', 'w') as fw:
            for batch_data in dataset.gen_score_batch():
                feed_dict_score = {self.input_mu: self.mus,
                                  self.input_sigma: self.sigmas,
                                  self.input_q: batch_data['q_test'],
                                  self.input_q_personal: batch_data['q_test_personal'],
                                  self.input_q_weight: batch_data['q_weight_test'],
                                  self.input_pos_d: batch_data['d1_test'],
                                  self.input_neg_d: batch_data['d2_test'],
                                  self.input_pos_d_personal: batch_data['d1_test_personal'],
                                  self.input_neg_d_personal: batch_data['d2_test_personal'],
                                  self.input_pos_f: batch_data['f1_test'],
                                  self.input_neg_f: batch_data['f2_test'],
                                  self.encoder_inputs_length: batch_data['q_len_test'],
                                  self.decoder_inputs: batch_data['next_q_test'],
                                  self.decoder_inputs_length: batch_data['next_q_len_test'],
                                  self.input_Y: batch_data['Y_test'],
                                  self.lambdas: batch_data['lambda_test'],
                                  self.input_mask_pos: self.gen_mask(batch_data['q_test'], batch_data['d1_test']),
                                  self.input_mask_neg: self.gen_mask(batch_data['q_test'], batch_data['d2_test']),
                                  self.atten_mask_q: self.gen_atten_mask(batch_data['q_test'], self.max_q_len),
                                  self.atten_mask_pos: self.gen_atten_mask(batch_data['d1_test'], self.max_d_len),
                                  self.atten_mask_neg: self.gen_atten_mask(batch_data['d2_test'], self.max_d_len),
                                  self.atten_mask_q_personal: self.gen_atten_mask(batch_data['q_test_personal'], self.max_q_len),
                                  self.atten_mask_pos_personal: self.gen_atten_mask(batch_data['d1_test_personal'], self.max_d_len),
                                  self.atten_mask_neg_personal: self.gen_atten_mask(batch_data['d2_test_personal'], self.max_d_len),
                                  self.pos_d_mask: self.gen_doc_mask(batch_data['d1_test']),
                                  self.neg_d_mask: self.gen_doc_mask(batch_data['d2_test']),}
                scores = self.sess.run(self.scores, feed_dict_score)
                #print("input_q: ", batch_data['q_test'][0], "input_d: ", batch_data['d1_test'][0])
                #print("testing scores: ", scores[0])
                print("scores: ", scores[0])
                evaluation.write_score(scores, batch_data['lines_test'], fw)
        with open('test_score.txt', 'r') as f:
           evaluation.evaluate(f)

    def online_test(self, dataset):  # 在线测试模型，随着用户输入新的查询，持续更新对应用户的模型
        testing_loss = 0.0
        testing_acc = 0.0
        testing_steps = 0
        fw = open('online_test.txt', 'w')
        test_lines = []
        for batch_data in dataset.gen_test_batch():
            feed_dict_test = {self.input_mu: self.mus,
                              self.input_sigma: self.sigmas,
                              self.input_q: batch_data['q_test'],
                              self.input_q_personal: batch_data['q_test_personal'],
                              self.input_q_weight: batch_data['q_weight_test'],
                              self.input_pos_d: batch_data['d1_test'],
                              self.input_neg_d: batch_data['d2_test'],
                              self.input_pos_d_personal: batch_data['d1_test_personal'],
                              self.input_neg_d_personal: batch_data['d2_test_personal'],
                              self.input_pos_f: batch_data['f1_test'],
                              self.input_neg_f: batch_data['f2_test'],
                              self.encoder_inputs_length: batch_data['q_len_test'],
                              self.decoder_inputs: batch_data['next_q_test'],
                              self.decoder_inputs_length: batch_data['next_q_len_test'],
                              self.input_Y: batch_data['Y_test'],
                              self.lambdas: batch_data['lambda_test'],
                              self.input_mask_pos: self.gen_mask(batch_data['q_test'], batch_data['d1_test']),
                              self.input_mask_neg: self.gen_mask(batch_data['q_test'], batch_data['d2_test']),
                              self.atten_mask_q: self.gen_atten_mask(batch_data['q_test'], self.max_q_len),
                              self.atten_mask_pos: self.gen_atten_mask(batch_data['d1_test'], self.max_d_len),
                              self.atten_mask_neg: self.gen_atten_mask(batch_data['d2_test'], self.max_d_len),
                              self.atten_mask_q_personal: self.gen_atten_mask(batch_data['q_test_personal'], self.max_q_len),
                              self.atten_mask_pos_personal: self.gen_atten_mask(batch_data['d1_test_personal'], self.max_d_len),
                              self.atten_mask_neg_personal: self.gen_atten_mask(batch_data['d2_test_personal'], self.max_d_len),
                              self.pos_d_mask: self.gen_doc_mask(batch_data['d1_test']),
                              self.neg_d_mask: self.gen_doc_mask(batch_data['d2_test']),}
            _, test_loss_, test_acc_, scores = self.sess.run([self.train_step, self.loss, self.accuracy, self.scores], feed_dict_test)
            for i in range(len(scores)):
                if batch_data['line1_test'][i] not in test_lines:
                    test_lines.append(batch_data['line1_test'][i])
                    fw.write(batch_data['line1_test'][i].rstrip('\n')+'\t'+str(scores[i][0])+'\n')
                if batch_data['line2_test'][i] not in test_lines:
                    test_lines.append(batch_data['line2_test'][i])
                    fw.write(batch_data['line2_test'][i].rstrip('\n')+'\t'+str(scores[i][1])+'\n')
            #print("input_q: ", batch_data['q_test'][0], "input_d: ", batch_data['d1_test'][0])
            #print("testing scores: ", scores[0])
            testing_loss += test_loss_
            testing_acc += test_acc_
            testing_steps += 1
        fw.close()
        self.logger.info("Average loss on test dataset: " + str(testing_loss / testing_steps))
        self.logger.info("Accuracy: " + str(testing_acc / testing_steps))

    def save(self, model_dir):
        """
        save the model into model_dir
        """
        self.saver.save(self.sess, model_dir)
        self.logger.info('Model saved in {}.'.format(model_dir))

    def restore(self, model_dir):
        """
        restore the model from the model dir
        """
        self.saver.restore(self.sess, model_dir)
        self.logger.info('Model restored from {}.'.format(model_dir))


def parse_args():
    parser = argparse.ArgumentParser("knrm")
    parser.add_argument('--train', action='store_true',
                        help='whether to train the model.')
    parser.add_argument('--test', action='store_true',
                        help='whether to test the model.')
    parser.add_argument('--online_test', action='store_true',
                        help='whether to test the model online.')
    parser.add_argument('--score', action='store_true',
                        help='whether to score every document and get the ranking.')
    parser.add_argument('--log_path', 
                        help='path of the logging file.')
    parser.add_argument('--model_dir', default='/home/jing_yao/learning/personalized_embedding/models/knrm',
                        help='path to store the trained models.')
    parser.add_argument('--restore_path', default='/home/jing_yao/learning/personalized_embedding/models/knrm_0',
                        help='path to restore the model.')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='learning rate to train the model.')
    parser.add_argument('--epsilon', default=1e-5, type=float,
                        help='epsilon for AdamOptimizer.')
    parser.add_argument('--num_epoch', default=5, type=int,
                        help='The number of training epochs.')
    parser.add_argument('--batch_size', default=200, type=int,
                        help='Batch size of the input data.')
    parser.add_argument('--n_bins', default=11, type=int,
                        help='The number of kernels.')
    parser.add_argument('--max_q_len', default=20, type=int,
                        help='Max length of queries.')
    parser.add_argument('--max_d_len', default=50, type=int,
                        help='Max length of documents.')
    parser.add_argument('--embedding_size', default=50, type=int,
                        help='size of the word embedding.')
    parser.add_argument('--vocabulary_size', default=124067, type=int,
                        help='size of the vocabulary.')
    parser.add_argument('--embedding_path',
                        help='path of the word embeddings.')
    parser.add_argument('--vocab_path', default='/home/jing_yao/learning/personalization/AOL_Data/vocab.dict',
                        help='path of the vocabulary.')
    parser.add_argument('--with_idf', default=False, type=bool,
                        help='whether take the idf weight into consideration.')
    parser.add_argument('--feature_size', default=110, type=int,
                        help='dimension of the SLTB features.')
    parser.add_argument('--doc_attention', action='store_true',
                        help='whether do self-attention on the document.')
    parser.add_argument('--hidden_size', default=100, type=int,
                        help='dimension of hidden state in LSTM.')
    parser.add_argument('--use_transformer', action='store_true',
                        help='use transformer or only multi-head self-attention in our model.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logger = logging.getLogger("knrm")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    logger.info("building the model...")
    model = KNRM(args)
    logger.info("loading the dataset...")
    dataset = Dataset(batch_size = args.batch_size,
                      num_epoch = args.num_epoch,
                      with_idf = args.with_idf,
                      max_q_len = args.max_q_len,
                      max_d_len = args.max_d_len,
                      feature_size = args.feature_size)
    if args.train:
        dataset.prepare_train_dataset()
        model.train(dataset)
        model.test(dataset)

    if args.test:
        dataset.prepare_train_dataset()
        model.restore(args.restore_path)
        model.test(dataset)

    if args.online_test:
        dataset.prepare_train_dataset()
        model.restore(args.restore_path)
        model.online_test(dataset)

    if args.score:
        dataset.prepare_score_dataset()
        model.restore(args.restore_path)
        evaluation = MAP()
        model.score(dataset, evaluation)


if __name__ == '__main__':
    main()
