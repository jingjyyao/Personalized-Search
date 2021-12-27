"""
数据集文件记录：(1)数据集所在路径：/home/jing_yao/learning/personalization/AOL_Data/（115）
(2)用户的查询日志：ValidQLSample_HF目录下，每个文件代表一个用户，每条记录包括(keywords, queryid, sessionid, date, url, SAT, urlrank0/1)，用\t分隔
"""

import os
import copy
import pickle
import random
import numpy as np
from time import time
from datetime import datetime

class Dataset:
    VALID_WORD2VEC = {'Deepwalk', 'GoogleNews', 'Wiki', 'QueryLog'}
    FEATURE_SIZE = {'Deepwalk':300, 'GoogleNews':300, 'Wiki':1000, 'QueryLog':300}
    def __init__(self, data_path='/home/jing_yao/learning_code/personalization/AOL_Data/',
                querylogfile='ValidQLSample_HF', word2vec='Wiki', limitation=10000000,
                max_query_num=300, min_session=3, max_doc_num=10, max_pair_num=20, other_feature_size=110,
                vector_path='/home/shuqi_lu/Projects/AOL_Data/', batch_size=200, feature_size=50):
        """
        初始化数据集中需要用到的参数
        """
        if word2vec not in Dataset.VALID_WORD2VEC:
            raise ValueError("word2vec not valid")
        self.data_path = data_path # 数据路径
        #self.in_path = os.path.join(data_path, querylogfile) # 输入数据的路径
        self.in_path = '/home/jing_yao/learning_code/personalization/AOL_Data/SampleUsers'
        #self.in_path = '/home/jing_yao/learning/personalization/AOL_Data/ValidQLSample_HF'
        self.filenames = sorted(os.listdir(self.in_path))
        self.query2vec_path = '/home/shuqi_lu/Projects/AOL_data/QueryVec/'
        self.doc2vec_path = '/home/shuqi_lu/Projects/AOL_data/DocVec/'
        self.feature_size = feature_size
        self.other_feature_size = other_feature_size
        self.limitation = limitation
        self.max_query_num = max_query_num
        self.min_session = min_session
        self.max_doc_num = max_doc_num
        self.max_pair_num = max_pair_num  # 作为历史而言，一个query下最多保留多少个文档对
        self.batch_size = batch_size
        self.cutoff_date = '2006-04-03 00:00:00'  # 之前的作为历史数据，之后的为实验数据
        self.train_date = '2006-05-16 00:00:00'  # 之前的作为训练集，之后的平均分为验证集和测试集

    
    def init_dict(self):
        self.query2id=pickle.load(open('/home/shuqi_lu/Projects/AOL_data/query2id.dict','rb'))
        self.url2id=pickle.load(open('/home/shuqi_lu/Projects/AOL_data/url2id2.dict','rb'))

    
    def transform_q(self, sentence):
        idx = self.query2id[sentence]
        return pickle.load(open(os.path.join(self.query2vec_path, str(idx)+'.pkl'), 'rb'))

    
    def transform_u(self, sentence):
        idx = self.url2id[sentence]
        return pickle.load(open(os.path.join(self.doc2vec_path, str(idx)+'.pkl'), 'rb'))


    def divide_dataset(self, filename):
        """
        统计每个用户的queryLog中query和session的数目，并判断这个用户的数据是否有效，舍弃数据少于一定程度的用户（需要一定的用户历史才能够建立可靠的用户画像）
        """
        self.session_sum = 0
        self.session_train = 0
        self.session_valid = 0
        self.session_test = 0
        query_sum = 0
        last_queryid = 0
        last_sessionid = 0

        with open(os.path.join(self.in_path, filename)) as fhand:
            for line in fhand:
                try:
                    line, features = line.strip().split('###')
                except:
                    line = line.strip()
                user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t') 
                queryid = sessionid + querytime + query

                if querytime < self.cutoff_date:
                    if queryid != last_queryid:
                        last_queryid = queryid
                        query_sum += 1
                else:
                    if sessionid != last_sessionid:
                        self.session_sum += 1
                        assert(last_queryid != queryid)
                        last_sessionid = sessionid
                    if queryid != last_queryid:
                        last_queryid = queryid

            if self.session_sum < self.min_session:
                return False
            #self.session_test = self.session_valid
            #self.session_valid //= 2
            #self.session_test -= self.session_valid
        return True


    def init_dataset(self):
        self.trainset = []
        self.validset = []
        self.testset = []


    def cal_delta(self, targets): # 即两个文档交换位置会对列表map带来的变化
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        total_num_rel = 0
        total_metric = 0.0
        for i in range(n_targets):
            if targets[i] == 1:
                total_num_rel += 1
                total_metric += total_num_rel / (i + 1.0)
        metric = (total_metric / total_num_rel) if total_num_rel > 0 else 0.0  # 原clicks状态下的map值
        num_rel_i = 0  # 截止位置i之前的相关文档数
        for i in range(n_targets):
            if targets[i] == 1:
                num_rel_i += 1
                num_rel_j = num_rel_i
                sub = num_rel_i / (i + 1.0)
                for j in range(i+1, n_targets):
                    if targets[j] == 1:
                        num_rel_j += 1
                        sub += 1 / (j + 1.0)
                    else:
                        add = (num_rel_j / (j + 1.0))
                        new_total_metric = total_metric + add - sub
                        new_metric = new_total_metric / total_num_rel
                        deltas[i, j] = new_metric - metric

            else: # 位置i的文档不相关
                num_rel_j = num_rel_i
                add = (num_rel_i + 1) / (i + 1.0) # 如果位置i更换为相关的，map会增加值

                for j in range(i + 1, n_targets):
                    if targets[j] == 1:
                        sub = (num_rel_j + 1) / (j + 1.0)  # 把位置j的文档换成位置i的文档
                        new_total_metric = total_metric + add - sub
                        new_metric = new_total_metric / total_num_rel
                        deltas[i, j] = new_metric - metric  # 表示位置j的文档换到i之后map的增值
                        num_rel_j += 1
                        add += 1 / (j + 1.0) # 位置越靠后调整到前面增加的map越多
        return deltas


    def prepare_pairs(self, query):
        temp_query = copy.deepcopy(query)
        doc_label = temp_query['label']
        lbds = self.cal_delta(doc_label)
        doc_list = np.zeros((self.max_doc_num, self.feature_size))
        doc_list[:min(self.max_doc_num, len(doc_label))] = temp_query['doc_feature'][:min(self.max_doc_num, len(doc_label))]
        feature_list = np.zeros((self.max_doc_num, self.other_feature_size))
        feature_list[:min(self.max_doc_num, len(doc_label))] = temp_query['other_feature'][:min(self.max_doc_num, len(doc_label))]

        pairs = {'query_feature': temp_query['query_feature'],
                 'history': temp_query['history'],
                 'history_len': temp_query['history_len'],
                 'cut_flags': temp_query['cut_flags'],
                 'last_query': temp_query['last_query'],
                 'doc_list': doc_list,
                 'feature_list': feature_list,
                 'doc_label': doc_label,
                 'doc_pairs': [],
                 'feature_pairs': [],
                 'pair_labels': [], # 记录pair的正负 +1, -1
                 'action_labels': [],  # 记录应该选第几个action 0,1
                 'pair_indexs': [],  # 记录构成pair的doc在列表中的index
                 'pair_confidence': [],  # 记录构成的每一对pair有多高的置信度
                 'click_labels': [], # 记录一对pair中文档的点击情况
        }
        for i in range(min(self.max_doc_num, len(doc_label))-1):
            for j in range(i+1, min(self.max_doc_num, len(doc_label))):  # 考虑列表中所有有意义的pair
                if len(pairs['doc_pairs']) >= self.batch_size: # 最大一个batch
                    continue
                if doc_label[i] == doc_label[j]: # 不构成pair
                    continue
                doc_pair = np.zeros((2, self.feature_size))
                feature_pair = np.zeros((2, self.other_feature_size))
                doc_pair[0] = temp_query['doc_feature'][i]
                doc_pair[1] = temp_query['doc_feature'][j]
                feature_pair[0] = temp_query['other_feature'][i]
                feature_pair[1] = temp_query['other_feature'][j]
                pairs['doc_pairs'].append(doc_pair)
                pairs['feature_pairs'].append(feature_pair)
                pairs['click_labels'].append([doc_label[i], doc_label[j]])
                if doc_label[i] > doc_label[j]: # 1/0，若j是紧跟i下一个，置信度为1.5，否则置信度为1
                    pairs['pair_labels'].append([1,0])
                    pairs['action_labels'].append(0)
                    if j == i+1: # j紧跟i
                        pairs['pair_confidence'].append(1.5)
                    else:
                        pairs['pair_confidence'].append(1)
                elif doc_label[i] < doc_label[j]: # 0/1，置信度为2，这种pair是可信度最高的
                    pairs['pair_labels'].append([0,1])
                    pairs['action_labels'].append(1) # 只有两种选择
                    pairs['pair_confidence'].append(2)
                # else: # 0/0或1/1    # 这部分pair是否需要计算在内呢
                #     pairs['pair_labels'].append(0)
                #     pairs['action_labels'].append(1)
                #     pairs['pair_confidence'].append(0)
                pairs['pair_indexs'].append((i, j))
        return pairs


    def prepare_query(self, session_count, query):
        if session_count < self.session_train:
            self.trainset.append(query)
        elif session_count < self.session_train + self.session_valid:
            self.validset.append(query)
        else:
            self.testset.append(query)


    def prepare_session(self, session, label): # 以session为单位
        if label == 'train':
            self.trainset.append(session)
        elif label == 'valid':
            self.validset.append(session)
        else:
            self.testset.append(session)


    def create_pairs(self, query):
        """
            query中包含查询中所有的url和对应的label，需要构成pair作为查询历史进行存储
            每个query下最多生成max_query_num个pair，如果点击的文档数>=4.限制每个点击文档构成的pair不超过5个
        """
        urls, labels = query['urls'], query['labels']
        pairs = []
        if not 1 in labels:  # 没有点击文档
            for url in urls:
                pairs.append(('-', [url, url], (0, 0)))
                if len(pairs) >= self.max_pair_num:
                    break
        elif not 0 in labels: # 只有点击文档
            for url in urls:
                pairs.append(('+', [url, url], (0, 0)))
                if len(pairs) >= self.max_pair_num:
                    break
        else:
            limit = 6  # 注意通常情况下只有5个候选文档
            if sum(labels) >= 4:   # 如果点击的文档数>=4.限制每个点击文档构成的pair不超过5个
                limit = 2
            lbds = self.cal_delta(labels)
            for i in range(len(labels)-1):
                count = 0
                for j in range(i+1, len(labels)):
                    if labels[i] == labels[j]:
                        continue
                    count += 1
                    if count >= limit:
                        break
                    if labels[i] == 1 and labels[j] == 0:  #1在前，0在后
                        pairs.append((-lbds[i,j], [urls[i], urls[j]], (i, j)))
                    else:
                        pairs.append((lbds[i,j], [urls[j], urls[i]], (i, j)))
                    if len(pairs) >= self.max_pair_num:
                        break
                if len(pairs) >= self.max_pair_num:
                    break
        return pairs


    # 先把数据全部读入内存，然后再一一用于训练,暂时先不考虑shuffle的情况，就一一读入
    def prepare_dataset(self):
        """
        把数据集全部读入内存，然后再一一用于训练
        """
        if not hasattr(self, 'trainset'):
            self.init_dataset()
        if not hasattr(self, 'queryid'):
            self.init_dict()
        count = 0
        for filename in self.filenames:
            count += 1
            print("processing: %d/%d" % (count, len(self.filenames)))
            if not self.divide_dataset(filename): # 判断这个用户的数据是否有效
                continue
            query_count, session_count = -1, -1 # 表示在序号，从0开始
            last_queryid, last_sessionid = 0, 0
            one_query = {'doc_feature':[], 'label':[], 'position':[], 'other_feature':[]}
            last_query = {'urls': [], 'labels': []}
            history = []
            cut_flags = []
            session = []
            key = 0
            last_querytime = 0
            clicked_url_init = np.zeros((2*self.feature_size), dtype=np.float64)
            fr = open(os.path.join(self.in_path, filename))
            for line in fr:
                try:
                    line, features = line.strip().split('###')
                    features = [float(item) for item in features.split('\t')]
                    if np.isnan(np.array(features)).sum():  # 除去包含nan的feature
                        continue
                except:
                    line = line.strip()
                
                user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t') 
                queryid = sessionid + querytime + query

                if queryid != last_queryid:  # 开始一个新的query
                    queryvec = self.transform_q(query)
                if querytime < self.cutoff_date: # 之前的查询作为用户的历史查询
                    if queryid != last_queryid:
                        query_count += 1
                        if len(last_query['urls']) > 0:  # 上一个查询的全部记录
                            pairs = self.create_pairs(last_query) # 其中pair有可能是有正有负，也有可能不构成pair只有正或只有负
                            history_pairs = np.zeros((len(pairs), 3*self.feature_size+1))
                            #print(pairs)
                            for index, (label, pair, _) in enumerate(pairs): # 用label标记这个pair是+还是-，还是(+,-)
                                history_pairs[index][:self.feature_size] = last_query['query']
                                if label == '+': # pair中只有正例
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                elif label == '-': # pair中只有负例
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                else:
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[1])
                                    history_pairs[index][-1] = label
                            history.append(history_pairs)
                        last_query = {'urls': [], 'labels': []}
                        last_query['query'] = queryvec

                        if sessionid != last_sessionid:
                            cut_flags.append([1.0, 0.0])
                            #history.append(np.append(np.append(queryvec, clicked_url_init), [1.0, 0.0])) # 标记session的起始
                            if last_queryid != 0:
                                cut_flags[query_count-1][-1] = 1.0
                                #history[query_count-1][-1] = 1.0 # 不是第一组查询
                            last_sessionid = sessionid
                        else:
                            cut_flags.append([0.0, 0.0])
                            #history.append(np.append(np.append(queryvec, clicked_url_init), [0.0, 0.0])) #中间的任意一次查询
                        last_queryid = queryid
                        last_querytime = querytime

                    last_query['urls'].append(url)
                    last_query['labels'].append(int(sat))
                    #if int(sat) == 1:
                    #    key = 1
                    #    history[-1][self.feature_size: 2*self.feature_size] += self.transform_u(url.lower()) # 记录相关文档向量
                else:
                    if queryid != last_queryid: # 开始一个新的query
                        query_count += 1
                        if len(last_query['urls']) > 0:  # 上一个查询的全部记录
                            pairs = self.create_pairs(last_query) # 其中pair有可能是有正有负，也有可能不构成pair只有正或只有负
                            history_pairs = np.zeros((len(pairs), 3*self.feature_size+1))
                            #print(pairs)
                            for index, (label, pair, _) in enumerate(pairs): # 用label标记这个pair是+还是-，还是(+,-)
                                history_pairs[index][:self.feature_size] = last_query['query']
                                if label == '+': # pair中只有正例
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                elif label == '-': # pair中只有负例
                                    #print("history: ", history[-1])
                                    #print("url: ", self.transform_u(pair[0].lower()))
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                else:
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[1])
                                    history_pairs[index][-1] = label
                            history.append(history_pairs)
                        last_query = {'urls': [], 'labels': []}
                        last_query['query'] = queryvec

                        if len(one_query['doc_feature'])>1 and key == 1 and len(one_query['history'])>0: # 把上一个query的信息储存起来
                            session.append(one_query)
                            #self.prepare_query(session_count, one_query)
                        one_query = {'doc_feature':[], 'label':[], 'position':[], 'other_feature':[]}
                        key = 0
                        one_query['last_query'] = pairs
                        one_query['query_feature'] = queryvec
                        one_query['history'] = copy.copy(history)
                        one_query['cut_flags'] = copy.copy(cut_flags)
                        one_query['history_len'] = min(query_count, 300)  # 这个地方一定要注意，否则的话personalization根本就model不出来啊
                        if sessionid != last_sessionid:
                            if session != []:
                                if last_querytime < self.train_date:
                                    flag = 'train'
                                elif len(session[0]['doc_feature']) == 50:
                                    flag = 'test'
                                else:
                                    flag = 'valid'
                                self.prepare_session(session, flag)
                                session = []
                            session_count += 1
                            last_sessionid = sessionid
                            cut_flags.append([1.0, 0.0])
                            #history.append(np.append(np.append(queryvec, clicked_url_init), [1.0, 0.0])) # 标记session的起始
                            if last_queryid != 0:
                                cut_flags[query_count-1][-1] = 1.0
                                #history[query_count-1][-1] = 1.0 # 不是第一组查询
                            last_sessionid = sessionid
                        else:
                            cut_flags.append([0.0, 0.0])
                        last_queryid = queryid
                        last_querytime = querytime
                    one_query['doc_feature'].append(self.transform_u(url))
                    one_query['label'].append(int(sat))
                    one_query['position'].append(int(urlrank))
                    one_query['other_feature'].append(features)

                    last_query['urls'].append(url)
                    last_query['labels'].append(int(sat))
                    
                    if int(sat) == 1:
                        key = 1
                        #history[-1][self.feature_size: 2*self.feature_size] += self.transform_u(url.lower())
            if len(one_query['doc_feature'])>1 and key == 1 and len(one_query['history'])>0:
                session.append(one_query)
            if len(session) > 0:
                if last_querytime < self.train_date:
                    flag = 'train'
                elif len(session[0]['doc_feature']) == 50:
                    flag = 'test'
                else:
                    flag = 'valid'
                self.prepare_session(session, flag)
            #if len(one_query['doc_feature'])>1 and key == 1:  # 注意最后一个查询
            #    self.prepare_query(session_count, one_query)
            fr.close()
            # if len(self.validset) > 50:
            #    break
        print('Successfully prepared the dataset, trainset: {}, validset: {}, testset: {}'.format(len(self.trainset), len(self.validset), len(self.testset)))


    def dataset_iter(self, setname):
        if setname == 'train':
            indices = np.arange(len(self.trainset))
            np.random.shuffle(indices)
            for sid in indices:
                yield self.padding_data(self.trainset[sid])
        elif setname == 'valid':
            for query in self.validset:
                yield self.padding_data(query)
        elif setname == 'test':
            for query in self.testset:
                yield self.padding_data(query)


    def padding_data(self, session):
        """
        padding the history to be the same length
        """
        temp_session = []
        for query in session:
            temp_query = self.prepare_pairs(query)
            history = np.zeros((self.max_query_num, self.max_pair_num, 3*self.feature_size+1), dtype=np.float64)
            for index, his_query in enumerate(temp_query['history'][-self.max_query_num:]):
                history[index][:len(his_query)] = his_query
            temp_query['history'] = history
            cut_flags = np.zeros((self.max_query_num, 2), dtype=np.float64)
            cut_flags[:min(self.max_query_num, len(temp_query['cut_flags']))] = temp_query['cut_flags'][-self.max_query_num:]
            temp_query['cut_flags'] = cut_flags
            temp_session.append(temp_query)
        return temp_session


    # generate a batch of training data
    def gen_batch(self, replay_buffer, batch_size):
        batch_data = {'history': [],
                      'history_len': [],
                      'cut_flags': [],
                      'query_feature': [],
                      'doc_lists': [],
                      'feature_lists': [],
                      'doc_pairs': [],
                      'feature_pairs': [],
                      'labels': [],
                      'values': [],
                      'click_labels': []}
        max_action_num = 0
        for step in random.sample(replay_buffer, batch_size):
            history, cut_flags, query_feature, doc_list, feature_list, doc_pair, feature_pair, click_labels = step['state']
            batch_data['history'].append(history)
            batch_data['history_len'].append(step['history_len'])
            batch_data['cut_flags'].append(cut_flags)
            batch_data['query_feature'].append(query_feature)
            batch_data['doc_lists'].append(doc_list)
            batch_data['feature_lists'].append(feature_list)
            batch_data['doc_pairs'].append(doc_pair)
            batch_data['feature_pairs'].append(feature_pair)
            batch_data['labels'].append(step['action'])
            batch_data['values'].append(step['value'])
            batch_data['click_labels'].append(click_labels)
        return batch_data

    # generate a batch of online testing data
    def gen_test_batch(self, replay_buffer, batch_size):
        batch_data = {'history': [],
                      'history_len': [],
                      'query_feature': [],
                      'doc_lists': [],
                      'feature_lists': [],
                      'doc_pairs': [],
                      'feature_pairs': [],
                      'labels': [],
                      'values': [],
                      'click_labels': []}
        max_action_num = 0
        for step in replay_buffer[-batch_size:]:
            history, query_feature, doc_list, feature_list, doc_pair, feature_pair, click_labels = step['state']
            batch_data['history'].append(history)
            batch_data['history_len'].append(step['history_len'])
            batch_data['query_feature'].append(query_feature)
            batch_data['doc_lists'].append(doc_list)
            batch_data['feature_lists'].append(feature_list)
            batch_data['doc_pairs'].append(doc_pair)
            batch_data['feature_pairs'].append(feature_pair)
            batch_data['labels'].append(step['action'])
            batch_data['values'].append(step['value'])
            batch_data['click_labels'].append(click_labels)
        return batch_data


    def prepare_score_dataset(self):
        self.testset = []
        if not hasattr(self, 'queryid'):
            self.init_dict()
        count = 0
        for filename in self.filenames:
            count += 1
            print("processing %d/%d" % (count, len(self.filenames)))
            if not self.divide_dataset(filename): # 判断这个用户的数据是否有效
                continue
            query_count, session_count = -1, -1 # 表示在序号，从0开始
            last_queryid, last_sessionid = 0, 0
            session_count_all = 0
            one_query = {'doc_feature':[], 'label':[], 'position':[], 'other_feature':[], 'lines_test':[]}
            last_query = {'urls': [], 'labels': []}
            history = []
            cut_flags = []
            last_querytime = 0
            clicked_url_init = np.zeros((2*self.feature_size), dtype=np.float64)
            fr = open(os.path.join(self.in_path, filename))
            for line in fr:
                try:
                    line, features = line.strip().split('###')
                    features = [float(item) for item in features.split('\t')]
                    if np.isnan(np.array(features)).sum():  # 除去包含nan的feature
                        continue
                except:
                    line = line.strip()
                
                user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t') 
                queryid = sessionid + querytime + query

                if sessionid != last_sessionid:
                    session_count_all += 1
                if queryid != last_queryid:  # 开始一个新的query
                    queryvec = self.transform_q(query)
                if querytime < self.cutoff_date: # 之前的查询作为用户的历史查询
                    if queryid != last_queryid:
                        query_count += 1
                        if len(last_query['urls']) > 0:  # 上一个查询的全部记录
                            pairs = self.create_pairs(last_query) # 其中pair有可能是有正有负，也有可能不构成pair只有正或只有负
                            history_pairs = np.zeros((len(pairs), 3*self.feature_size+1))
                            #print(pairs)
                            for index, (label, pair, _) in enumerate(pairs): # 用label标记这个pair是+还是-，还是(+,-)
                                history_pairs[index][:self.feature_size] = last_query['query']
                                if label == '+': # pair中只有正例
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                elif label == '-': # pair中只有负例
                                    #print("history: ", history[-1])
                                    #print("url: ", self.transform_u(pair[0].lower()))
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                else:
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[1])
                                    history_pairs[index][-1] = label
                            history.append(history_pairs)
                        last_query = {'urls': [], 'labels': []}
                        last_query['query'] = queryvec

                        if sessionid != last_sessionid:
                            cut_flags.append([1.0, 0.0])
                            #history.append(np.append(np.append(queryvec, clicked_url_init), [1.0, 0.0])) # 标记session的起始
                            if last_queryid != 0:
                                cut_flags[query_count-1][-1] = 1.0
                                #history[query_count-1][-1] = 1.0 # 不是第一组查询
                            last_sessionid = sessionid
                        else:
                            cut_flags.append([0.0, 0.0])
                        last_queryid = queryid
                        last_querytime = querytime
                    last_query['urls'].append(url)
                    last_query['labels'].append(int(sat))
                    #if int(sat) == 1:
                        #history[-1][self.feature_size: 2*self.feature_size] += self.transform_u(url.lower()) # 记录相关文档向量
                else:
                    if queryid != last_queryid: # 开始一个新的query
                        query_count += 1
                        if len(last_query['urls']) > 0:  # 上一个查询的全部记录
                            pairs = self.create_pairs(last_query) # 其中pair有可能是有正有负，也有可能不构成pair只有正或只有负
                            history_pairs = np.zeros((len(pairs), 3*self.feature_size+1))
                            #print(pairs)
                            for index, (label, pair, _) in enumerate(pairs): # 用label标记这个pair是+还是-，还是(+,-)
                                history_pairs[index][:self.feature_size] = last_query['query']
                                if label == '+': # pair中只有正例
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                elif label == '-': # pair中只有负例
                                    #print("history: ", history[-1])
                                    #print("url: ", self.transform_u(pair[0].lower()))
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][-1] = 0.1
                                else:
                                    history_pairs[index][self.feature_size: 2*self.feature_size] += self.transform_u(pair[0])
                                    history_pairs[index][2*self.feature_size: 3*self.feature_size] += self.transform_u(pair[1])
                                    history_pairs[index][-1] = label
                            history.append(history_pairs)
                        last_query = {'urls': [], 'labels': []}
                        last_query['query'] = queryvec

                        if len(one_query['doc_feature'])>0 and len(one_query['history'])>0: # 把上一个query的信息储存起来
                            if last_querytime >= self.train_date:
                                self.testset.append(one_query)
                        one_query = {'doc_feature':[], 'label':[], 'position':[], 'other_feature':[], 'lines_test':[]}
                        key = 0
                        one_query['last_query'] = pairs
                        one_query['query_feature'] = queryvec
                        one_query['history'] = copy.copy(history)
                        one_query['cut_flags'] = copy.copy(cut_flags)
                        one_query['history_len'] = min(query_count, 300)  # 这个地方一定要注意，否则的话personalization根本就model不出来啊
                        if sessionid != last_sessionid:
                            session_count += 1
                            last_sessionid = sessionid
                            cut_flags.append([1.0, 0.0])
                            #history.append(np.append(np.append(queryvec, clicked_url_init), [1.0, 0.0])) # 标记session的起始
                            if last_queryid != 0:
                                cut_flags[query_count-1][-1] = 1.0
                                #history[query_count-1][-1] = 1.0 # 不是第一组查询
                            last_sessionid = sessionid
                        else:
                            cut_flags.append([0.0, 0.0])
                        last_queryid = queryid
                        last_querytime = querytime
                    one_query['doc_feature'].append(self.transform_u(url))
                    one_query['label'].append(int(sat))
                    one_query['position'].append(int(urlrank))
                    one_query['other_feature'].append(features)
                    one_query['lines_test'].append(line)

                    last_query['urls'].append(url)
                    last_query['labels'].append(int(sat))

                    #if int(sat) == 1:
                    #    history[-1][self.feature_size: 2*self.feature_size] += self.transform_u(url.lower())
            if len(one_query['label'])!=0 and last_querytime >= self.train_date and len(one_query['history']) > 0:  # 注意最后一个查询、最后一个session的处理
                self.testset.append(one_query)
            fr.close()
            #if len(self.testset) > 20:
            #    break
        print('Successfully prepared the scoring testset: {}'.format(len(self.testset)))

    
    def gen_score_batch(self, batch_size):
        batch_data = {'history': [],
                      'history_len': [],
                      'cut_flags': [],
                      'query_feature': [],
                      'doc_lists': [],
                      'feature_lists': [],
                      'doc_pairs': [],
                      'feature_pairs': [],
                      'labels': [],
                      'values': [],
                      'lines_test': [],
                      'click_labels': []}
        for query in self.testset:
            for didx in range(len(query['label'])):
                batch_data['query_feature'].append(query['query_feature'])
                history = np.zeros((self.max_query_num, self.max_pair_num, 3*self.feature_size+1), dtype=np.float64)
                for index, his_query in enumerate(query['history'][-self.max_query_num:]):
                    history[index][:len(his_query)] = his_query
                batch_data['history'].append(history)
                batch_data['history_len'].append(query['history_len'])
                
                cut_flags = np.zeros((self.max_query_num, 2), dtype=np.float64)
                cut_flags[:min(self.max_query_num, len(query['cut_flags']))] = query['cut_flags'][-self.max_query_num:]
                batch_data['cut_flags'].append(cut_flags)
                
                doc_list = np.zeros((self.max_doc_num, self.feature_size))
                feature_list = np.zeros((self.max_doc_num, self.other_feature_size))
                batch_data['doc_lists'].append(doc_list)
                batch_data['feature_lists'].append(feature_list)
                doc_pair = np.zeros((2, self.feature_size))
                feature_pair = np.zeros((2, self.other_feature_size))
                doc_pair[0] = query['doc_feature'][didx]
                feature_pair[0] = query['other_feature'][didx]
                batch_data['doc_pairs'].append(doc_pair)
                batch_data['feature_pairs'].append(feature_pair)
                batch_data['labels'].append(0)
                batch_data['values'].append(0)
                batch_data['lines_test'].append(query['lines_test'][didx])
                batch_data['click_labels'].append([1,1])

                if len(batch_data['values']) == batch_size:
                    yield batch_data
                    batch_data = {'history': [],
                                  'history_len': [],
                                  'cut_flags': [],
                                  'query_feature': [],
                                  'doc_lists': [],
                                  'feature_lists': [],
                                  'doc_pairs': [],
                                  'feature_pairs': [],
                                  'labels': [],
                                  'values': [],
                                  'lines_test': [],
                                  'click_labels': []}


    def repeated_query(self):
        if not hasattr(self, 'queryid'):
            self.init_dict()
        repeat_query = []
        count = 0
        for filename in self.filenames:
            count += 1
            print("processing: %d/%d" % (count, len(self.filenames)))
            if not self.divide_dataset(filename):
                continue
            last_queryid, last_sessionid = 0, 0
            session_count = 0
            fr = open(os.path.join(self.in_path, filename))
            for line in fr:
                try:
                    line, features = line.strip().split('###')
                    features = [float(item) for item in features.split('\t')]
                    if np.isnan(np.array(features)).sum():  # 除去包含nan的feature
                        continue
                except:
                    line = line.strip()
                
                user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t') 
                queryid = sessionid + querytime + query
                query = user + ' ' + query
                if queryid != last_queryid: # 完成上一个查询
                    last_queryid = queryid
                    if query not in repeat_query:
                        repeat_query.append(query)
                if querytime >= self.train_date:
                    break
        with open('repeat_query','wb') as fw:
            pickle.dump(repeat_query, fw)