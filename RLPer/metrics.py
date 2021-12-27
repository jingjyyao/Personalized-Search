"""
This file implements multiple metrics for personalization search, including
MAP, MRR, P@1, Avg.Click, #Better, P-Improve
"""
import os
import pickle
import numpy as np

class Metric(object):
    """
    Base metric class.
    Subclasses must override score() and can optionally override other methods.
    """
    def __init__(self):
        print("Generating the entropy dictionary.")
        with open('/home/jing_yao/learning_code/personalization/AOL_Data/entropy.dict', 'rb') as fr:
            self.entropy = pickle.load(fr)

    def score(self, clicks):
        raise NotImplementedError

    def rerank(self, scores, lines):
        # rerank the lists according to the scores
        idxs = list(zip(*sorted(enumerate(scores), key=lambda x:x[1], reverse=True)))[0]
        final_lines = []
        for idx in idxs:
            final_lines.append(lines[idx])
        return final_lines

    def evaluate(self, filehandle):
        """
        evaluate queries in the filehadle, each line is the same as the training dataset.
        """
        last_queryid = 0
        f = open(self.__class__.__name__+'.txt', 'w')
        mscore1, mscore2 = 0.0, 0.0
        nquery = 0.0
        clicks = []
        scores = []
        for line in filehandle:
            user, sessionid, querytime, query, url, title, sat, urlrank, _, score = line.strip().split('\t') 
            queryid = user + sessionid + querytime + query

            if queryid != last_queryid: # 表示一个query结束了
                if len(clicks) != 0 and len(clicks) != 1 and len(clicks) == 50:
                    score1 = self.score(clicks)
                    score2 = self.score(self.rerank(scores, clicks))
                    if score1 != -1:
                        nquery += 1
                        mscore1 += score1
                        mscore2 += score2
                    f.write(last_queryid+'\t'+str(self.entropy[query])+'\t'+
                        '\t'+str(score1)+'\t'+str(score2)+'\n')
                clicks = []
                scores = []
                last_queryid = queryid
            clicks.append(sat)
            scores.append(float(score))
        if len(clicks) != 0 and len(clicks) != 1 and len(clicks)==50:
            score1 = self.score(clicks)
            score2 = self.score(self.rerank(scores, clicks))
            if score1 != -1:
                nquery += 1
                mscore1 += score1
                mscore2 += score2
            f.write(last_queryid+'\t'+str(self.entropy[query])+'\t'+
                '\t'+str(score1)+'\t'+str(score2)+'\n')    
        f.close()  
        print("The "+self.__class__.__name__+" of original ranking is {}.".format(mscore1/nquery))
        print("The "+self.__class__.__name__+" of new ranking is {}.".format(mscore2/nquery))

    def write_score(self, scores, lines, filehandle):
        assert(len(scores)==len(lines))
        for i in range(len(scores)):
            filehandle.write(lines[i].rstrip('\n')+'\t'+str(scores[i])+'\n')


class MAP(Metric):
    # 平均正确率(Average Precision)：对不同召回率点上的正确率进行平均。
    def  __init__(self, cutoff='1'):
        super(MAP, self).__init__()
        self.cutoff = cutoff

    def score(self, clicks):
        num_rel = 0
        total_prec = 0.0
        for i in range(len(clicks)):
            if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
                num_rel += 1
                total_prec += num_rel / (i + 1.0)
        return (total_prec / num_rel) if num_rel > 0 else -1


class MRR(Metric):
    # 第一个正确答案位置的倒数，取平均值
    def __init__(self, cutoff='1'):
        super(MRR, self).__init__()
        self.cutoff = cutoff

    def score(self, clicks):
        num_rel = 0
        total_prec = 0.0
        for i in range(len(clicks)):
            if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
                num_rel = 1
                total_prec = 1.0 / (i+1)
                break
        return total_prec if num_rel > 0 else -1


class Precision(Metric):
    # precision@k: 计算前k个文档中有几个是正确文档  # 本文中这种计算方式合理嘛？
    def __init__(self, cutoff='1', k=1):
        super(Precision, self).__init__()
        self.cutoff = cutoff
        self.k = k

    def score(self, clicks):
        prec_in = 0.0
        prec_out = 0.0
        for i in range(len(clicks)):
            if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
                if i+1 <= self.k:
                    prec_in = 1
                    break
                else:
                    prec_out = 1
        if prec_in > 0:
            return 1
        else:
            return 0 if prec_out > 0 else -1


class AvePosition(Metric):
    # 平均位置：所有相关文档的位置的平均值
    def __init__(self, cutoff='1'):
        super(AvePosition, self).__init__()
        self.cutoff = cutoff

    def score(self, clicks):
        position = 0.0
        nclick = 0.0
        for i in range(len(clicks)):
            if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
                position += i+1
                nclick += 1
        return (position/nclick) if nclick > 0 else -1


class Pimprove(Metric):
    def __init__(self, cutoff='1'):
        super(Pimprove, self).__init__()
        self.cutoff = cutoff

    def score(self, clicks, scores):
        up, down, pairs = 0, 0, 0
        for i in range(len(clicks)):
            if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
                for j in range(0, i): # the above un-clicked
                    if clicks[j] == 0: # count the original inverse document pairs
                        pairs += 1 
                    if clicks[j]==0 and scores[i]>scores[j]: # rank the sat-click before the above un-clicked
                        up+=1
            
                if i+1<len(clicks):
                    j=i+1
                    if clicks[j]==0:
                        pairs+=1
                    if clicks[j]==0 and scores[i]<scores[j]: # rank the next un-click before the sat-click
                        down+=1            
        return up,down,pairs


    def evaluate(self, filehandle):
        last_queryid = 0
        nup = 0.0
        ndown = 0.0
        npairs = 0.0
        clicks = []
        scores = []
        for line in filehandle:
            user, sessionid, querytime, query, url, title, sat, urlrank, _, score = line.strip().split('\t') 
            queryid = user + sessionid + querytime + query

            if queryid != last_queryid:
                if len(clicks) != 0 and len(clicks) != 1:
                    up,down,pairs=self.score(clicks,scores)
                    nup += up
                    ndown += down
                    npairs += pairs
                clicks = []
                scores = []
                last_queryid = queryid
            clicks.append(int(sat))
            scores.append(float(score))
        if len(clicks) != 0 and len(clicks) != 1:
            up,down,pairs=self.score(clicks,scores)
            nup += up
            ndown += down
            npairs += pairs

        print("The number of better rankings is {}.".format(nup))
        print("The number of worse rankings is {}.".format(ndown))
        print("The Pgain is {}.".format((nup-ndown)/(npairs)))


class SessionMAP(Metric):
    def __init__(self, cutoff='1'):
        super(SessionMAP, self).__init__()
        self.cutoff = cutoff


    def score(self, clicks):
        num_rel = 0
        total_prec = 0.0
        for i in range(len(clicks)):
            if clicks[i] == self.cutoff or clicks[i] == int(self.cutoff):
                num_rel += 1
                total_prec += num_rel / (i + 1.0)
        return (total_prec / num_rel) if num_rel > 0 else -1


    def evaluate(self, filehandle):
        last_sessionid = 0
        last_queryid = 0
        session_scores1, session_scores2 = [], []
        query_scores1, query_scores2 = [], []
        mscore1, mscore2 = 0.0, 0.0
        nquery = 0.0
        clicks = []
        scores = []
        for line in filehandle:
            user, sessionid, querytime, query, url, title, sat, urlrank, _, score = line.strip().split('\t') 
            queryid = user + sessionid + querytime + query

            if queryid != last_queryid: # 表示一个query结束了
                if len(clicks) != 0:
                    score1 = self.score(clicks)
                    score2 = self.score(self.rerank(scores, clicks))
                    if score1 != -1:
                        query_scores1.append(score1)
                        query_scores2.append(score2)
                    clicks = []
                    scores = []
                if sessionid != last_sessionid:
                    if len(query_scores1) > 0:
                        session_scores1.append(np.mean(query_scores1))
                        session_scores2.append(np.mean(query_scores2))
                    query_scores1, query_scores2 = [], []
                    last_sessionid = sessionid
                last_queryid = queryid
            clicks.append(sat)
            scores.append(float(score))

        if len(clicks) != 0:  # 最后一个query和最后一个session
            score1 = self.score(clicks)
            score2 = self.score(self.rerank(scores, clicks))
            if score1 != -1:
                query_scores1.append(score1)
                query_scores2.append(score2)
        if len(query_scores1) > 0:
            session_scores1.append(np.mean(query_scores1))
            session_scores2.append(np.mean(query_scores2))

        print("The SessionMAP of original ranking is {}.".format(np.mean(session_scores1)))
        print("The SessionMAP of new ranking is {}.".format(np.mean(session_scores2)))
