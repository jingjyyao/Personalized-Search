import os
import types
import random
import logging
import argparse
import numpy as np
from metrics import *
import tensorflow as tf
from dataset import Dataset
# from dataset_bing import Dataset
from HGRUCell import HGRUCell
from itertools import permutations

"""
 This model is pairwise, each step corresponds to determining a document pair,
 and a session is an episode
"""
class RLPer(object):
    def __init__(self, args):
        
        # logging
        self.logger = logging.getLogger('rlper')

        # basic config
        self.batch_size = args.batch_size
        self.state_size = args.state_size # size of the user profile vector
        self.short_state_size = args.short_state_size # size of the long-term user profile vector
        self.long_state_size = args.long_state_size  # size of the short-term user profile vector
        self.atten_hidden_size = args.atten_hidden_size  # hidden size of the attention layer
        self.hidden_unit_num = args.hidden_unit_num  # hidden size of the fully connected layer in profile
        self.max_query_num = args.max_query_num # max num ber of query in history
        self.feature_size = args.feature_size # size of the feature vector of the query and document
        self.other_feature_size = args.other_feature_size # number of the other features 
        self.perm_size = args.perm_size # number of the document for permutation
        self.max_doc_num = args.max_doc_num # max number of docs for a query
        self.max_pair_num = args.max_pair_num  # max number of doc pair under a query
        self.discount_factor = args.discount_factor
        self.optim_type = args.optim_type
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon # value of epsilon for imitation
        self.epsilon_decay = args.epsilon_decay
        self.replay_memory_size = args.replay_memory_size
        self.encode_type = args.encode_type # approaches for encoding 

        # reward
        self.reward_type = args.reward_type
        self.MRR = MRR()
        self.evaluation = MAP()
        
        # session info
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # run on cpu
        self.sess = tf.Session()

        # build the model and parameter setting
        self.user_profile()  # build user profile
        self.actor()  # tell the relationships of the document pair
        self.create_optimizer()
        
        # save info
        self.model_dir = args.model_dir
        self.saver = tf.train.Saver()

        # summary info
        self.merged = tf.summary.merge_all()
        self.log_dir = args.log_dir
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')

        # initialize the model
        self.sess.run(tf.global_variables_initializer())


    def dynamic_atten(self, outputs):
        """
        dynamically model the user profile with attention mechanism
        """
        with tf.variable_scope('AttentionLayer'):
            Attn_q = tf.layers.dense(inputs=self.q, units=self.atten_hidden_size, activation=tf.nn.sigmoid)
            Attn_q = tf.layers.dense(inputs=self.q, units=self.feature_size, activation=tf.nn.sigmoid)
        ratios = tf.reduce_sum(self.history_q*tf.expand_dims(Attn_q, 1), 2)
        mask = tf.sequence_mask(self.history_len, self.max_query_num, dtype=tf.float64)
        self.a = tf.exp(ratios)/tf.expand_dims(tf.reduce_sum(tf.exp(ratios)*mask, 1), 1)
        atten_final_state = tf.reduce_sum(outputs*tf.expand_dims(self.a, 2)*tf.expand_dims(mask, 2), 1)
        return atten_final_state


    def user_profile(self):
        """
        Model the history and generate the user profile, hierarchically and dynamically
        """
        self.history = tf.placeholder(tf.float64, [None, self.max_query_num, self.max_pair_num, 3*self.feature_size+1]) # 查询历史，每一条对应一个矩阵，矩阵中的每一条对应一个文档对
        self.cut_flags = tf.placeholder(tf.float64, [None, self.max_query_num, 2])  # 0，1，用于表示session的起始位置
        self.history_len = tf.placeholder(tf.float64, [None])
        self.q = tf.placeholder(tf.float64, [None, self.feature_size])  # current query
        with tf.variable_scope('user_profile'):
            if self.encode_type == 'dense':
                history = tf.layers.dense(inputs=self.history, units=self.feature_size, activation=tf.nn.relu)
                history = tf.layers.dense(inputs=tf.transpose(history, perm=[0,1,3,2]), units=1, activation=tf.nn.relu)
            if self.encode_type == 'self-attention':
                # 先把history映射到attention维的空间
                history = tf.layers.dense(inputs=self.history, units=self.feature_size, activation=tf.nn.relu)
                # 然后计算每个pair对其他pair的权重
                attention = tf.nn.softmax(tf.matmul(history, tf.transpose(history, perm=[0,1,3,2])), -1)
                # 然后求加权和作为每个位置的结果
                history = tf.matmul(attention, history)
                history = tf.layers.dense(inputs=tf.transpose(history, perm=[0,1,3,2]), units=1, activation=tf.nn.relu)

            history = tf.concat([tf.squeeze(history, -1), self.cut_flags], -1)
            cell = HGRUCell(num_units = self.short_state_size + self.long_state_size,  # hierarchical RNN
                            short_state_size = self.short_state_size,
                            long_state_size = self.long_state_size)
            init_state = tf.get_variable('init_state',[1, self.state_size], dtype=tf.float64)
            init_state = tf.tile(init_state, [self.batch_size, 1])
            output, self.final_state = tf.nn.dynamic_rnn(cell=cell, inputs=history, sequence_length=self.history_len, initial_state=init_state)#, initial_state=init_state)

            self.history_q, self.history_d, _ = tf.split(self.history, num_or_size_splits=[self.feature_size, 2*self.feature_size, 1], axis=3)
            self.history_q = self.history_q[:, :, 0, :]
            self.final_state =  self.dynamic_atten(output)
            self.short_final_state, self.long_final_state = tf.split(self.final_state, num_or_size_splits=[self.short_state_size, self.long_state_size], axis=1)


    def actor(self):
        """
        Build the agent actor
        """
        self.doc_lists = tf.placeholder(tf.float64, [self.batch_size, self.max_doc_num, self.feature_size])  # high-level MDP
        self.feature_lists = tf.placeholder(tf.float64, [self.batch_size, self.max_doc_num, self.other_feature_size])
        self.doc_pairs = tf.placeholder(tf.float64, [self.batch_size, 2, self.feature_size])
        self.feature_pairs = tf.placeholder(tf.float64, [self.batch_size, 2, self.other_feature_size])
        self.labels = tf.placeholder(tf.int32, [self.batch_size])  # 文档对的标签
        self.values = tf.placeholder(tf.float64, [self.batch_size])  # reward
        self.click_labels = tf.placeholder(tf.float64, [self.batch_size, 2]) # 标注pair中的文档是否有点击

        with tf.variable_scope('MatchingUserProfileDoc'):
            short_user_profile = tf.layers.dense(inputs=self.short_final_state, units=self.hidden_unit_num, activation=tf.nn.sigmoid)
            short_user_profile = tf.layers.dense(inputs=short_user_profile, units=self.feature_size, activation=tf.nn.sigmoid)
            list_short_user_profile = tf.tile(tf.reshape(short_user_profile, (-1, 1, self.feature_size)), [1, self.max_doc_num, 1])
            short_user_profile = tf.tile(tf.reshape(short_user_profile, (-1, 1, self.feature_size)), [1, 2, 1])
            print("short_user_profile: ", short_user_profile.shape)
            
            long_user_profile = tf.layers.dense(inputs=self.long_final_state, units=self.hidden_unit_num, activation=tf.nn.sigmoid)
            long_user_profile = tf.layers.dense(inputs=long_user_profile, units=self.feature_size, activation=tf.nn.sigmoid)
            list_long_user_profile = tf.tile(tf.reshape(long_user_profile, (-1, 1, self.feature_size)), [1, self.max_doc_num, 1])
            long_user_profile = tf.tile(tf.reshape(long_user_profile, (-1, 1, self.feature_size)), [1, 2, 1])
            print("long_user_profile: ", long_user_profile.shape)
            
            
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_pairs), axis=-1))
            list_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_lists), axis=-1))

            short_norm = tf.sqrt(tf.reduce_sum(tf.square(short_user_profile), axis=-1))
            long_norm = tf.sqrt(tf.reduce_sum(tf.square(long_user_profile), axis=-1))
            list_short_norm = tf.sqrt(tf.reduce_sum(tf.square(list_short_user_profile), axis=-1))
            list_long_norm = tf.sqrt(tf.reduce_sum(tf.square(list_long_user_profile), axis=-1))

            short_scores = tf.reduce_sum(short_user_profile * self.doc_pairs, -1) / (doc_norm * short_norm + 1)
            long_scores = tf.reduce_sum(long_user_profile * self.doc_pairs, -1) / (doc_norm * long_norm + 1)

            list_short_scores = tf.reduce_sum(list_short_user_profile * self.doc_lists, -1) / (list_norm * list_short_norm + 1)
            list_long_scores = tf.reduce_sum(list_long_user_profile * self.doc_lists, -1) / (list_norm * list_long_norm + 1)

            feature_scores = tf.layers.dense(inputs=self.feature_pairs, units=1, activation=tf.nn.tanh)
            list_feature_scores = tf.layers.dense(inputs=self.feature_lists, units=1, activation=tf.nn.tanh)


        final_scores = tf.concat([tf.concat([tf.expand_dims(short_scores, 2), tf.expand_dims(long_scores, 2)], -1), feature_scores], -1)
        self.final_scores = tf.squeeze(tf.layers.dense(inputs=final_scores, units=1, activation=None), -1)

        list_final_scores = tf.concat([tf.concat([tf.expand_dims(list_short_scores, 2), tf.expand_dims(list_long_scores, 2)], -1), list_feature_scores], -1)
        self.list_final_scores = tf.squeeze(tf.layers.dense(inputs=list_final_scores, units=1, activation=None), -1)

        # compute the action probability
        #print("shape: ", (self.final_scores[:,0]-self.final_scores[:,1]).shape)
        #pair_scores = tf.concat([tf.concat([tf.reshape(self.final_scores[:,0]-self.final_scores[:,1], (self.batch_size, 1)), tf.zeros((self.batch_size, 1), dtype=tf.float64)], -1), tf.reshape(self.final_scores[:,1]-self.final_scores[:,0], (self.batch_size, 1))], -1) # (batchsize,3)
        pair_scores = tf.concat([tf.reshape(self.final_scores[:,0]-self.final_scores[:,1], (self.batch_size, 1)), tf.reshape(self.final_scores[:,1]-self.final_scores[:,0], (self.batch_size, 1))], -1) # 只考虑两种情况
        self.pair_scores = tf.nn.softmax(pair_scores)
        self.prediction = tf.argmax(self.pair_scores, axis=1)

        with tf.variable_scope("loss"):
            #self.click_score = tf.reduce_sum(-self.final_scores * self.click_labels, axis = -1) # 最大化点击文档的分数，最小化其负值
            self.neg_log = tf.reduce_sum(-tf.log(self.pair_scores) * tf.cast(tf.one_hot(indices = self.labels, depth=2), dtype=tf.float64), axis=1)
            self.loss = tf.reduce_mean(self.neg_log * self.values)# + tf.reduce_mean(self.click_score)
            tf.summary.scalar("loss", self.loss)


    def create_optimizer(self):
        """
        Select the training algorithm
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        with tf.variable_scope("train"):
            self.train_op = self.optimizer.minimize(self.loss)


    def feedback_his(self, history, history_len, last_query, last_lambdas, data):
        """
         add the user's real-time feedback to the history
        """
        click_init = np.zeros((self.max_pair_num, 2*self.feature_size+1))
        for index, (label, pair, position) in enumerate(last_query):
            if position == (0, 0):
                return
            try:
                click_init[index][:self.feature_size] = data.transform_u(pair[0])
                click_init[index][self.feature_size: 2*self.feature_size] = data.transform_u(pair[1])
                click_init[index][-1] = np.abs(last_lambdas[position])
            except:
                click_init[index][:self.feature_size] = data.transform_u(pair[0])
                click_init[index][self.feature_size: 2*self.feature_size] = data.transform_u(pair[1])
                click_init[index][-1] = 0.1
            history[history_len-1, :, self.feature_size: 3*self.feature_size+1] = click_init


    def genEpisode(self, data, epochs):
        """
        generate episodes for Monte Carlo Algorithms, an episode corresponds to a session, a step to a pair inner this session
        """
        maxacc = 0
        batch_idx = 0
        data.prepare_dataset()
        self.logger.info('Successfully prepared the dataset, trainset: {}, validset: {}, testset: {}'.format(len(data.trainset), len(data.validset), len(data.testset)))
       
        action2label = {0: 1, 1: -1}
        for epoch in range(1, epochs + 1):
            #if epoch <= 2:
            #    self.epsilon = 1  # 前两个epoch先采用监督学习的方式进行训练后面在增强
            trainset = data.dataset_iter('train')
            self.replay_buffer = []
            epoch_rewards = []
            losses = []
            for idx, session in enumerate(trainset):
                if idx%2000==0 and idx>0:
                    print("processing training step/epoch: %d/%d" % (idx, epoch))
                    self.logger.info("Mean of rewards before %d/%d: %f" % (idx, epoch, np.mean(epoch_rewards)))
                episode = []
                last_lambdas = dict()

                for qidx, query in enumerate(session):
                    query_feature = query['query_feature']
                    history_len = query['history_len']
                    history = query['history']    # 这里的history应该根据上一次的点击进行实时的修改
                    cut_flags = query['cut_flags']  # 拼接在history后面
                    if qidx > 0 and len(last_lambdas) > 0:  # 不是当前session中的第一个查询
                        self.feedback_his(history, history_len, query['last_query'], last_lambdas, data)
                    # print("history: ", history)

                    doc_list = query['doc_list']
                    feature_list = query['feature_list']
                    doc_label = query['doc_label']
                    doc_pairs = query['doc_pairs']
                    feature_pairs = query['feature_pairs']
                    pair_labels = query['pair_labels']
                    pair_indexs = query['pair_indexs']
                    action_labels = query['action_labels']
                    pair_confidence = query['pair_confidence']
                    click_labels = query['click_labels']
                    if len(doc_pairs) <= 0:
                        last_lambdas = dict()
                        continue

                    self.epsilon_list = np.zeros((self.batch_size, 2))
                    for lidx, label in enumerate(query['action_labels']):
                        self.epsilon_list[lidx][label] = 1


                    # first step: Sample the episodes for Monte Carlo
                    feed_dict = {self.history: [history] * self.batch_size,
                                 self.history_len: [history_len] * self.batch_size,
                                 self.cut_flags: [cut_flags] * self.batch_size,
                                 self.q: [query_feature] * self.batch_size,
                                 self.doc_lists: [doc_list] * self.batch_size,
                                 self.feature_lists: [feature_list] * self.batch_size,
                                 self.doc_pairs: doc_pairs + [doc_pairs[-1]] * (self.batch_size - len(doc_pairs)),
                                 self.feature_pairs: feature_pairs + [feature_pairs[-1]] * (self.batch_size - len(doc_pairs)),
                                 self.labels: [0] * self.batch_size,
                                 self.values: [0] * self.batch_size,
                                 self.click_labels: [[1,1]] * self.batch_size}
                    list_scores, scores = self.sess.run([self.list_final_scores, self.pair_scores], feed_dict=feed_dict)
                    #print("scores: ", scores[0])

                    # 根据模型返回的排序结果计算pair的lambda值作为reward
                    score_labels = []
                    for i in range(min(self.max_doc_num, len(doc_label))):
                        score_labels.append([i, list_scores[0][i], doc_label[i]])
                    score_labels = sorted(score_labels, key=lambda x: x[1], reverse=True) # 根据模型的打分得到新的文档列表
                    if self.reward_type == 'MAP':
                        lambdas = data.cal_delta(np.array(score_labels)[:,2])
                    else:
                        lambdas = self.cal_reward(np.array(score_labels)[:,2])
                    new_pair_lambdas = dict()
                    for i in range(min(self.max_doc_num, len(doc_label))-1):
                        for j in range(i+1, min(self.max_doc_num, len(doc_label))):
                            if score_labels[i][-1] == score_labels[j][-1]:
                                continue
                            new_pair_lambdas[(score_labels[i][0], score_labels[j][0])] = -lambdas[i,j]
                            new_pair_lambdas[(score_labels[j][0], score_labels[i][0])] = lambdas[i,j]                       
                            # assert(lambdas[i,j] != 0)
                    last_lambdas = new_pair_lambdas

                    policy = (1 - self.epsilon) * scores + self.epsilon * self.epsilon_list  # combined policy
                        
                    for i in range(len(doc_pairs)):
                        action = np.random.choice(2, 1, p=policy[i].squeeze())[0]
                        # if self.reward_type == 'MAP':
                        reward = action2label[action] * new_pair_lambdas[pair_indexs[i]] # pair_index存储的是一个doc pair，(id1, id2)
                        # else:
                        # reward = action2label[action] * self.getReward(pair_indexs[i], score_labels)
                        # reward = action2label[action] * new_pair_lambdas[pair_indexs[i]]
                        if random.random() > 0.9999:
                            print("pair: ", pair_labels[i], "label: ", action_labels[i], "action: ", action, "reward: ", reward, "confidece: ", pair_confidence[i])
                        epoch_rewards.append(reward)
                        episode.append({'history_len': history_len, 'state': (history, cut_flags, query_feature, doc_list, feature_list, doc_pairs[i], feature_pairs[i], click_labels[i]), 'action': action, 'reward': reward, 'value': reward})
                
                for step in np.arange(len(episode)-2, -1, -1):  # compute accumulative reward
                    episode[step]['value'] += self.discount_factor * episode[step+1]['value']
                
                for step in episode:  # mmethod of DQN
                    if len(self.replay_buffer) == self.replay_memory_size:
                        self.replay_buffer.pop(0)
                    self.replay_buffer.append(step)

                if len(self.replay_buffer) >= self.batch_size:# and sample_add > self.batch_size/2:
                    batch_data = data.gen_batch(self.replay_buffer, self.batch_size)
                    feed_dict = {self.history: batch_data['history'],
                                 self.history_len: batch_data['history_len'],
                                 self.cut_flags: batch_data['cut_flags'],
                                 self.q: batch_data['query_feature'],
                                 self.doc_lists: batch_data['doc_lists'],
                                 self.feature_lists: batch_data['feature_lists'],
                                 self.doc_pairs: batch_data['doc_pairs'],
                                 self.feature_pairs: batch_data['feature_pairs'],
                                 self.labels: batch_data['labels'],
                                 self.values: batch_data['values'],
                                 self.click_labels: batch_data['click_labels']}
                    # print("values: ", batch_data['values'])
                    summary, loss, _ = self.sess.run([self.merged, self.loss, self.train_op], feed_dict = feed_dict)
                    losses.append(loss)
                    self.train_writer.add_summary(summary, batch_idx)

                    if batch_idx % 2000 == 0 and batch_idx > 0:
                        self.logger.info("Mean loss from %d-%d/%d: %f" % (batch_idx-1000, batch_idx, epoch, np.mean(np.array(losses))))
                        self.logger.info("Current epsilon: %f" % self.epsilon)
                        losses = []
                    batch_idx += 1
                    # if batch_idx % 10000 == 0:
                    #     self.scoring(data, self.evaluation, 'test_score_'+str(batch_idx)+'.txt')

            self.logger.info("Mean rewards in epoch %d: %f" % (epoch, np.mean(epoch_rewards)))
            self.epsilon *= self.epsilon_decay  # the episode decays

            self.save(self.model_dir + '_' + str(epoch))

            # print("Evaluating the model...")
            # accuracy, eval_reward = self.Evaluate(data, 'valid')
            # self.logger.info("Evaluating result in epoch%d: accuracy %f, eval reward %f" % (epoch, accuracy, eval_reward))        
        

    def cal_reward(self, targets):  # 即两个文档交换位置会对列表的metric带来的变化
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        old_pairs = 0
        for i in range(len(targets)):
            if targets[i] == 1:
                for j in range(0, i): # the above un-clicked
                    if targets[j] == 0: # count the original inverse document pairs
                        old_pairs += 1                         
                if i+1<len(targets):
                    j=i+1
                    if targets[j]==0:
                        old_pairs+=1

        for i in range(n_targets):
            for j in range(i+1, n_targets):
                new_labels = [label for label in targets]
                new_labels[i], new_labels[j] = new_labels[j], new_labels[i]
                if self.reward_type == 'MRR':
                    delta = self.MRR.score(new_labels) - self.MRR.score(targets)
                elif self.reward_type == 'Inverse_Pair':
                    new_pairs = 0
                    for k in range(len(new_labels)):
                        if new_labels[k] == 1:
                            for t in range(0, k): # the above un-clicked
                                if new_labels[t] == 0: # count the original inverse document pairs
                                    new_pairs += 1                         
                            if k+1<len(new_labels):
                                t=k+1
                                if new_labels[t]==0:
                                    new_pairs+=1

                    delta = -(new_pairs - old_pairs)
                deltas[i,j] = delta  # 如果i=1,j=0, delta[i,j]为负值，因此lambda应该为-delta
        return deltas


    def getReward(self, pair_indexs, score_labels):
        index1, index2 = pair_indexs
        for iid, index in enumerate(score_labels): # 在score label中找到对应的两个文档
            if index[0] == index1:
                temp1 = iid
            if index[0] == index2:
                temp2 = iid
        old_labels = np.array(score_labels)[:, 2]
        new_labels = np.array(score_labels)[:, 2]
        new_labels[temp1], new_labels[temp2] = new_labels[temp2], new_labels[temp1] # 交换两个文档在列表中的位置

        if self.reward_type == 'MRR':
            if temp1 < temp2:
                return self.MRR.score(old_labels) - self.MRR.score(new_labels)

        elif self.reward_type == 'Inverse_Pair':
            old_pair, new_pair = 0, 0
            for i in range(len(old_labels)):
                if old_labels[i] == 1:
                    for j in range(0, i): # the above un-clicked
                        if old_labels[j] == 0: # count the original inverse document pairs
                            old_pairs += 1                         
                    if i+1<len(clicks):
                        j=i+1
                        if old_labels[j]==0:
                            old_pairs+=1

            for i in range(len(new_labels)):
                if new_labels[i] == 1:
                    for j in range(0, i): # the above un-clicked
                        if new_labels[j] == 0: # count the original inverse document pairs
                            new_pairs += 1                         
                    if i+1<len(clicks):
                        j=i+1
                        if new_labels[j]==0:
                            new_pairs+=1

            return new_pairs - old_pairs


    def Evaluate(self, data, setname):
        """
        evaluate the validset/testset
        """
        total_pair, right_pair = 0, 0
        eval_rewards = []
        action2label = {0: 1, 1: -1}
        validset = data.dataset_iter('valid')
        for idx, session in enumerate(validset):
            if idx%1000==0:
                print("processing validing step: %d" % idx)
            last_lambdas = dict()
            for qidx, query in enumerate(session):
                history = query['history']
                query_feature = query['query_feature']
                history_len = query['history_len']
                cut_flags = query['cut_flags']
                if qidx > 0 and len(last_lambdas) > 0:
                    self.feedback_his(history, history_len, query['last_query'], last_lambdas, data)

                doc_list = query['doc_list']
                feature_list = query['feature_list']
                doc_label = query['doc_label']
                doc_pairs = query['doc_pairs']
                feature_pairs = query['feature_pairs']
                pair_labels = query['pair_labels']
                action_labels = query['action_labels']
                pair_confidence = query['pair_confidence']
                pair_indexs = query['pair_indexs']
                if len(doc_pairs) <= 0:
                    last_lambdas = dict()
                    continue

                feed_dict = {self.history: [history] * self.batch_size,
                             self.history_len: [history_len] * self.batch_size,
                             self.cut_flags: [cut_flags] * self.batch_size,
                             self.q: [query_feature] * self.batch_size,
                             self.doc_lists: [doc_list] * self.batch_size,
                             self.feature_lists: [feature_list] * self.batch_size,
                             self.doc_pairs: doc_pairs + [doc_pairs[-1]] * (self.batch_size - len(doc_pairs)),
                             self.feature_pairs: feature_pairs + [feature_pairs[-1]] * (self.batch_size - len(doc_pairs)),
                             self.labels: [0] * self.batch_size,
                             self.values: [0] * self.batch_size,
                             self.click_labels: [[1,1]] *self.batch_size}
                predictions, list_scores = self.sess.run([self.prediction, self.list_final_scores], feed_dict=feed_dict)

                score_labels = []
                for i in range(min(self.max_doc_num, len(doc_label))):
                    score_labels.append([i, list_scores[0][i], doc_label[i]])
                score_labels = sorted(score_labels, key=lambda x: x[1], reverse=True)
                lambdas = data.cal_delta(np.array(score_labels)[:,2])
                new_pair_lambdas = dict()
                for i in range(min(self.max_doc_num, len(doc_label))-1):
                    for j in range(i+1, min(self.max_doc_num, len(doc_label))):
                        if score_labels[i][-1] == score_labels[j][-1]:
                            continue
                        new_pair_lambdas[(score_labels[i][0], score_labels[j][0])] = -lambdas[i,j]
                        new_pair_lambdas[(score_labels[j][0], score_labels[i][0])] = -lambdas[j,i] # 注意这个地方，新的排序可能会打乱原来的先后顺序，导致lambda的顺序出现问题                        
                last_lambdas = new_pair_lambdas

                for pidx in range(len(pair_labels)):
                    if new_pair_lambdas[pair_indexs[pidx]] != 0:
                        eval_rewards.append(action2label[predictions[pidx]] * new_pair_lambdas[pair_indexs[pidx]])
                        if predictions[pidx] == action_labels[pidx]:
                            right_pair += 1
                        total_pair += 1
        print("Accuracy: %f, Eval Rewards: %f" % (right_pair/total_pair, np.mean(eval_rewards)))
        return right_pair / total_pair, np.mean(eval_rewards)


    def scoring(self, data, evaluation, resultFile='test_score.txt'):
        data.prepare_score_dataset()
        with open(resultFile, 'w') as f:
            for batch_data in data.gen_score_batch(self.batch_size):
                feed_dict = {self.history: batch_data['history'],
                             self.history_len: batch_data['history_len'],
                             self.cut_flags: batch_data['cut_flags'],
                             self.q: batch_data['query_feature'],
                             self.doc_lists: batch_data['doc_lists'],
                             self.feature_lists: batch_data['feature_lists'],
                             self.doc_pairs: batch_data['doc_pairs'],
                             self.feature_pairs: batch_data['feature_pairs'],
                             self.labels: batch_data['labels'],
                             self.values: batch_data['values']}
                scores = self.sess.run(self.final_scores, feed_dict=feed_dict)
                evaluation.write_score(scores[:, 0], batch_data['lines_test'], f)
        with open(resultFile, 'r') as f:
            evaluation.evaluate(f)


    def save(self, model_dir):
        """
        saves the model into model_dir
        """
        self.saver.save(self.sess, model_dir)
        self.logger.info('Model saved in {}.'.format(model_dir))

    
    def restore(self, model_dir):
        """
        restore the model from the model_dir
        """
        self.saver.restore(self.sess, model_dir)
        self.logger.info('Model restored from {}.'.format(model_dir))


def parse_args():
    """
    Parses command line arguments
    """
    parser = argparse.ArgumentParser('Reinforcement Learning Personalization search')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--valid', action='store_true',
                        help='valid the model')
    parser.add_argument('--test', action='store_true',
                        help='test the model')
    parser.add_argument('--score', action='store_true',
                        help='score the model')
    parser.add_argument('--restore', action='store_true',
                        help='restore the model from the restore path')
    parser.add_argument('--restore_dir',
                        help='path for restoring the model.')
    parser.add_argument('--state_size', type=int, default=900,
                        help='length of the hidden state vector of the user profile')
    parser.add_argument('--short_state_size', type=int, default=300,
                        help='length of the short state vector')
    parser.add_argument('--long_state_size', type=int, default=600,
                        help='length of the long state vector')
    parser.add_argument('--max_query_num', type=int, default=300,
                        help='max number of queries in user history')
    parser.add_argument('--feature_size', type=int, default=50,
                        help='size of feature for queryvector or docvector')
    parser.add_argument('--other_feature_size', type=int, default=98,
                        help='number of other features')
    parser.add_argument('--perm_size', type=int, default=3,
                        help='number of document need for permutation')
    parser.add_argument('--discount_factor', type=float, default=0.8,
                        help='discount factor for reinforcement learning')
    parser.add_argument('--data_path', default='/home/songwei_ge/personalize/ProcessedData/',
                        help='Directory for the processed data')
    parser.add_argument('--querylogfile', default='ValidQLSample_HF',
                        help='File storing the query log')
    parser.add_argument('--word2vec', default='Wiki',
                        help='class of word2vec')
    parser.add_argument('--min_session', type=int, default=4,
                        help='minimum session for a valid user')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epoch for training')
    parser.add_argument('--log_path',
                        help='path for logging file')
    parser.add_argument('--log_dir', default='./log',
                        help='dir to store the log of summary operation.')
    parser.add_argument('--model_dir', default='../model',
                        help='dir to store the trained model')
    parser.add_argument('--optim_type', default='adam',
                        help='type of optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--max_doc_num', type=int, default=10,
                        help='max number of document under a query')
    parser.add_argument('--max_pair_num', type=int, default=20,
                        help='max number of document pairs under a query')
    parser.add_argument('--atten_hidden_size', type=int, default=300,
                        help='size of the hidden state in attention MLP')
    parser.add_argument('--hidden_unit_num', type=int, default=512,
                        help='size of the fully_connected layer in profile')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='epsilon, the additional value for imitation')
    parser.add_argument('--epsilon_decay', type=float, default=0.95,
                        help='factor for controlling the epsilon decay')
    parser.add_argument('--replay_memory_size', type=int, default=1000,
                        help='size of reply_buffer to store the history')
    parser.add_argument('--encode_type', default='dense',
                        help='approaches for encoding the document pairs matrix, including dense/self-attention/flatten')
    parser.add_argument('--reward_type', default='MAP',
                        help='compute reward for document pairs based on MAP/MRR/Inverse_Pair')
    return parser.parse_args()


def main():
    """
    run the whole model
    """
    args = parse_args()

    logger = logging.getLogger("rlper")
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
    print('Running with args : {}'.format(args))
    
    data = Dataset(data_path = args.data_path, 
                   querylogfile = args.querylogfile,
                   word2vec = args.word2vec,
                   max_query_num = args.max_query_num,
                   min_session = args.min_session,
                   max_doc_num = args.max_doc_num,
                   max_pair_num = args.max_pair_num,
                   other_feature_size = args.other_feature_size,
                   batch_size = args.batch_size)  
    model = RLPer(args)  

    if args.train:
        if args.restore:
            model.restore(args.restore_dir)
        model.genEpisode(data, args.epochs)

    if args.score:
        evaluation = MAP()
        model.restore(args.model_dir)
        model.scoring(data, evaluation)

if __name__ == '__main__':
    main()
 