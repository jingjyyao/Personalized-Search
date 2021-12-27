# 此程序实现用联邦学习来训练模型
import os
import copy
import time
import pickle
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from options import args_parser
from metrics import *
from localModel import LocalModel
from globalModel import GlobalModel
from dataset import Dataset


def average_weight(local_weights, count, total_batch):
    # weights_sum = local_weights[0]
    # for weight in local_weights[1:]:
    #     for ws, w in zip(weights_sum, weight):
    #         ws += w
    global_weight = []
    for weight in local_weights:
        global_weight.append(weight / count) #total_batch)  # global_weight为加权和
    return global_weight


def train(args):
    start_time = time.time()
    logger = logging.getLogger("peps")
    logger.info("building the global and local model...")
    global_model = GlobalModel(args)
    local_model = LocalModel(args)

    in_path = args.in_path
    all_users = sorted(os.listdir(in_path))
    users = []  # 只包括有效的用户数据
    dataset = Dataset(args, args.batch_size)
    for user in all_users:
        dataset.init_user(user, args.batch_size)
        if dataset.divide_dataset(user):
            users.append(user)
    print('Total number of valid users: ', len(users))


    # 检查一下users中是否和users_vocab中的用户有重叠
    valid_users = [user for user in local_model.users_vocab]
    print("number of valid users: ", len(valid_users))
    print("overlap users: ", len(set(users) & set(valid_users)))


    # copy global weights
    logger.info("getting the global model weights...")
    global_weight = global_model.get_global_weights()

    # Federated Average Training
    train_loss, train_accuracy = [], []
    valid_loss, valid_accuracy = [], []
    test_losses, test_accs = [], []
    for epoch in tqdm(range(args.num_epoch)):
        local_weights, local_losses, local_acc = [], [], []
        logger.info("\n Global training round: {}".format(epoch+1))
        m = max(int(args.fraction * len(users)), 1)
        idxs_users = np.random.choice(range(len(users)), m, replace=False)
        sampled_users = [users[idx] for idx in idxs_users]
        total_batch = 0
        batchnum_dict = {}

        # local_epoch = args.local_epoch  # 动态调整epoch
        # if (epoch+1)%20 == 0:
        #     local_epoch = max(local_epoch-2, 2)

        for idx, user in tqdm(enumerate(sampled_users)):
            local_model.init_local(args=args, batch_size=args.batch_size, user=user, global_weight=global_weight)
            weight, acc, loss, batch_count, sample_count = local_model.train()
            total_batch += batch_count
            batchnum_dict[batch_count] = batchnum_dict.get(batch_count, 0) + 1
            global_weight = weight  # 第一种情形下我们采用流式地训练方式
            local_losses.append(copy.deepcopy(loss))
            local_acc.append(acc)

        print("distribution of batches: ", batchnum_dict)

        train_loss.append(sum(local_losses) / len(local_losses))
        train_accuracy.append(sum(local_acc) / len(local_acc))

        logger.info('Average training states after {} global rounds.'.format(epoch+1))
        logger.info('Training loss: {}, training accuracy: {}'.format(train_loss[-1], train_accuracy[-1]))
        local_model.save(args.model_dir + '_' + str(epoch)) 

            
        #if not args.test_before_aggre:
        if epoch > 0 and epoch % 5 == 0:  # 测试太频繁会影响运行速度
            logger.info('calculate average testing accuracy over all users')
            list_acc, list_loss = [], []
            for idx, user in tqdm(enumerate(users)):
                local_model.init_local(args=args, batch_size=args.batch_size, user=user, global_weight=global_weight)
                acc, loss, steps = local_model.inference(setname='test')
                if steps > 0:
                    list_acc.append(acc)
                    list_loss.append(loss)
            logger.info('Average testing states after {} global rounds.'.format(epoch+1))
            logger.info('Test loss: {}, test accuracy: {}'.format(sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc))) 
            test_losses.append(sum(list_loss) / len(list_loss))
            test_accs.append(sum(list_acc) / len(list_acc))
            print("Test losses: ", test_losses)
            print("Test accs: ", test_accs) 
    print('\n Total time cost: ', time.time() - start_time)


def score(args):
    start_time = time.time()
    logger = logging.getLogger("peps")
    logger.info("building the global and local model...")
    global_model = GlobalModel(args)
    local_model = LocalModel(args)

    in_path = args.in_path
    all_users = sorted(os.listdir(in_path))
    users = []  # 只包括有效的用户数据
    dataset = Dataset(args, args.batch_size)
    for user in all_users:
        dataset.init_user(user, args.batch_size)
        if dataset.divide_dataset(user):
            users.append(user)
    print('Total number of valid users: ', len(users))

    global_model.restore(args.restore_path)
    global_weight = global_model.get_global_weights()
    
    fhand = open(args.score_file, 'w')
    evaluation = MAP()
    set_personal_embed = False
    for idx, user in tqdm(enumerate(users)):
        local_model.init_local(args=args, batch_size=args.batch_size, user=user, global_weight=global_weight)
        local_model.score(evaluation, fhand)
    fhand.close()
    with open(args.score_file, 'r') as fr:
        evaluation.evaluate(fr)

    print('\n Total time cost: ', time.time() - start_time)


if __name__ == '__main__':
    args = args_parser()
    logger = logging.getLogger("peps")
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
    
    if args.train:
        train(args) 

    if args.score:
        score(args) 