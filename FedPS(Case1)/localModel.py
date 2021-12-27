# 此程序实现了每个客户端本地更新模型
from PEPS import PEPS
from dataset import Dataset
import tensorflow as tf
import logging
import time
import pickle
import numpy as np


class LocalModel:
    def __init__(self, args):
        # logging info
        self.logger = logging.getLogger('peps')

        self.args = args

        # personal embeddings
        self.init_user_vocab()

        # build the model
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.graph.as_default():
            self.model = PEPS(args)
            self.sess.run(tf.global_variables_initializer())
            self.global_embedding = self.model.load_embedding(args.embedding_path, args.vocab_path, args.vocabulary_size)
            self.model.embeddings.load(self.global_embedding, self.sess)
            self.personal_embedding = self.sess.run(self.model.person_embeddings)
            self.saver = tf.train.Saver()
        self.dataset = Dataset(args=args, batch_size=args.batch_size) 

    def init_local(self, args, batch_size, user, global_weight, model_dir='model'):
        self.model_dir = model_dir
        self.user = user
        self.set_global_weights(global_weight)  # 用global weight进行初始化，包括global word embedding和ranking parameters
        # self.set_personal_embed()

        # the user's individual dataset
        self.dataset.init_user(user, batch_size)


    def init_user_vocab(self):
        self.vocab = pickle.load(open('/home/jing_yao/learning/personalization/AOL_Data/vocab.dict', 'rb'))
        self.person_vocab = pickle.load(open('/home/jing_yao/learning/personalization/AOL_Data/doc_w2v/person_vocab.dict', 'rb'))
        self.users_vocab = pickle.load(open('/home/jing_yao/learning/personalization/AOL_Data/doc_w2v/users_vocab.dict', 'rb'))
        self.users_word_embedding = pickle.load(open('/home/jing_yao/learning/personalization/AOL_Data/doc_w2v/users_word_embedding.dict', 'rb'))


    def update_users_embedding(self, user_word_embedding):
        if self.user in self.users_vocab:          
            vocab = self.users_vocab[self.user]
            for word, wordid in vocab.items():
                for i in range(self.args.embedding_size):
                    self.users_word_embedding[self.user][word][i] = user_word_embedding[wordid][i]


    def train(self):
        self.dataset.prepare_train_dataset()
        with self.graph.as_default():
            self.train_losses = []
            self.train_accuracies = []
            self.valid_losses = []
            self.valid_accuracies = []

            train_start_time = time.time()
            batch_count = 0
            for idx, epoch_data in enumerate(self.dataset.gen_epochs()):
                training_loss = 0.0
                training_acc = 0.0
                training_steps = 0.0
                valid_loss = 0.0
                valid_acc = 0.0
                valid_steps = 0.0
                for batch_data in epoch_data:
                    batch_start_time = time.time()
                    feed_dict_train = {self.model.input_mu: self.model.mus,
                                       self.model.input_sigma: self.model.sigmas,
                                       self.model.input_q: batch_data['q_train'],
                                       self.model.input_q_personal: batch_data['q_train_personal'],
                                       self.model.input_q_weight: batch_data['q_weight_train'],
                                       self.model.input_pos_d: batch_data['d1_train'],
                                       self.model.input_neg_d: batch_data['d2_train'],
                                       self.model.input_pos_f: batch_data['f1_train'],
                                       self.model.input_neg_f: batch_data['f2_train'],
                                       self.model.encoder_inputs_length: batch_data['q_len_train'],
                                       self.model.decoder_inputs: batch_data['next_q_train'],
                                       self.model.decoder_inputs_length: batch_data['next_q_len_train'],
                                       self.model.input_Y: batch_data['Y_train'],
                                       self.model.lambdas: batch_data['lambda_train'],
                                       self.model.input_mask_pos: self.model.gen_mask(batch_data['q_train'], batch_data['d1_train']),
                                       self.model.input_mask_neg: self.model.gen_mask(batch_data['q_train'], batch_data['d2_train']),
                                       self.model.atten_mask_q: self.model.gen_atten_mask(batch_data['q_train'], self.model.max_q_len),
                                       self.model.atten_mask_pos: self.model.gen_atten_mask(batch_data['d1_train'], self.model.max_d_len),
                                       self.model.atten_mask_neg: self.model.gen_atten_mask(batch_data['d2_train'], self.model.max_d_len),
                                       self.model.atten_mask_q_personal: self.model.gen_atten_mask(batch_data['q_train_personal'], self.model.max_q_len),
                                       self.model.pos_d_mask: self.model.gen_doc_mask(batch_data['d1_train']),
                                       self.model.neg_d_mask: self.model.gen_doc_mask(batch_data['d2_train']),}
                    train_loss_, train_acc_, _ = self.sess.run([self.model.loss, self.model.accuracy, self.model.train_step], feed_dict_train)
                    training_loss += train_loss_
                    training_acc += train_acc_
                    training_steps += 1
                    batch_count += 1

                    # if training_steps % 5 == 0:
                    #     print("\nBatch Loss: ", train_loss_, "\tAccuracy: ", train_acc_, "\tTime cost: ", time.time()-batch_start_time, "s.")
                    #     self.logger.info('training loss before {} step: {}.'.format(training_steps, training_loss / training_steps))


                self.train_losses.append(training_loss / max(training_steps, 1))
                self.train_accuracies.append(training_acc / max(training_steps, 1))

            local_weight = self.sess.run(tf.trainable_variables())
            # user_word_embedding = self.sess.run(self.model.person_embeddings)
        # self.update_users_embedding(user_word_embedding)
        return local_weight, sum(self.train_accuracies)/max(len(self.train_accuracies), 1), sum(self.train_losses)/max(len(self.train_losses), 1), batch_count, len(self.dataset.Y_train) #, sum(self.valid_losses)/max(len(self.valid_losses), 1)


    def inference(self, setname='train'):
        self.dataset.prepare_train_dataset()
        with self.graph.as_default():
            if setname == 'train':
                training_loss = 0.0
                training_acc = 0.0
                training_steps = 0
                for batch_data in self.dataset.gen_train_batch():
                    feed_dict_train = {self.model.input_mu: self.model.mus,
                                       self.model.input_sigma: self.model.sigmas,
                                       self.model.input_q: batch_data['q_train'],
                                       self.model.input_q_personal: batch_data['q_train_personal'],
                                       self.model.input_q_weight: batch_data['q_weight_train'],
                                       self.model.input_pos_d: batch_data['d1_train'],
                                       self.model.input_neg_d: batch_data['d2_train'],
                                       self.model.input_pos_f: batch_data['f1_train'],
                                       self.model.input_neg_f: batch_data['f2_train'],
                                       self.model.encoder_inputs_length: batch_data['q_len_train'],
                                       self.model.decoder_inputs: batch_data['next_q_train'],
                                       self.model.decoder_inputs_length: batch_data['next_q_len_train'],
                                       self.model.input_Y: batch_data['Y_train'],
                                       self.model.lambdas: batch_data['lambda_train'],
                                       self.model.input_mask_pos: self.model.gen_mask(batch_data['q_train'], batch_data['d1_train']),
                                       self.model.input_mask_neg: self.model.gen_mask(batch_data['q_train'], batch_data['d2_train']),
                                       self.model.atten_mask_q: self.model.gen_atten_mask(batch_data['q_train'], self.model.max_q_len),
                                       self.model.atten_mask_pos: self.model.gen_atten_mask(batch_data['d1_train'], self.model.max_d_len),
                                       self.model.atten_mask_neg: self.model.gen_atten_mask(batch_data['d2_train'], self.model.max_d_len),
                                       self.model.atten_mask_q_personal: self.model.gen_atten_mask(batch_data['q_train_personal'], self.model.max_q_len),
                                       self.model.pos_d_mask: self.model.gen_doc_mask(batch_data['d1_train']),
                                       self.model.neg_d_mask: self.model.gen_doc_mask(batch_data['d2_train']),}
                    train_loss_, train_acc_, scores= self.sess.run([self.model.loss, self.model.accuracy, self.model.scores], feed_dict_train)
                    training_loss += train_loss_
                    training_acc += train_acc_
                    training_steps += 1
                return training_acc/max(training_steps, 1), training_loss/max(training_steps, 1), training_steps

            if setname == 'valid':
                validing_loss = 0.0
                validing_acc = 0.0
                validing_steps = 0
                for batch_data in self.dataset.gen_valid_batch():
                    feed_dict_valid = {self.model.input_mu: self.model.mus,
                                       self.model.input_sigma: self.model.sigmas,
                                       self.model.input_q: batch_data['q_valid'],
                                       self.model.input_q_personal: batch_data['q_valid_personal'],
                                       self.model.input_q_weight: batch_data['q_weight_valid'],
                                       self.model.input_pos_d: batch_data['d1_valid'],
                                       self.model.input_neg_d: batch_data['d2_valid'],
                                       self.model.input_pos_f: batch_data['f1_valid'],
                                       self.model.input_neg_f: batch_data['f2_valid'],
                                       self.model.encoder_inputs_length: batch_data['q_len_valid'],
                                       self.model.decoder_inputs: batch_data['next_q_valid'],
                                       self.model.decoder_inputs_length: batch_data['next_q_len_valid'],
                                       self.model.input_Y: batch_data['Y_valid'],
                                       self.model.lambdas: batch_data['lambda_valid'],
                                       self.model.input_mask_pos: self.model.gen_mask(batch_data['q_valid'], batch_data['d1_valid']),
                                       self.model.input_mask_neg: self.model.gen_mask(batch_data['q_valid'], batch_data['d2_valid']),
                                       self.model.atten_mask_q: self.model.gen_atten_mask(batch_data['q_valid'], self.model.max_q_len),
                                       self.model.atten_mask_pos: self.model.gen_atten_mask(batch_data['d1_valid'], self.model.max_d_len),
                                       self.model.atten_mask_neg: self.model.gen_atten_mask(batch_data['d2_valid'], self.model.max_d_len),
                                       self.model.atten_mask_q_personal: self.model.gen_atten_mask(batch_data['q_valid_personal'], self.model.max_q_len),
                                       self.model.pos_d_mask: self.model.gen_doc_mask(batch_data['d1_valid']),
                                       self.model.neg_d_mask: self.model.gen_doc_mask(batch_data['d2_valid']),}
                    valid_loss_, valid_acc_, scores= self.sess.run([self.model.loss, self.model.accuracy, self.model.scores], feed_dict_valid)
                    validing_loss += valid_loss_
                    validing_acc += valid_acc_
                    validing_steps += 1
                return validing_acc/max(validing_steps, 1), validing_loss/max(validing_steps, 1), validing_steps

            if setname == 'test':
                testing_loss = 0.0
                testing_acc = 0.0
                testing_steps = 0
                for batch_data in self.dataset.gen_test_batch():
                    feed_dict_test = {self.model.input_mu: self.model.mus,
                                       self.model.input_sigma: self.model.sigmas,
                                       self.model.input_q: batch_data['q_test'],
                                       self.model.input_q_personal: batch_data['q_test_personal'],
                                       self.model.input_q_weight: batch_data['q_weight_test'],
                                       self.model.input_pos_d: batch_data['d1_test'],
                                       self.model.input_neg_d: batch_data['d2_test'],
                                       self.model.input_pos_f: batch_data['f1_test'],
                                       self.model.input_neg_f: batch_data['f2_test'],
                                       self.model.encoder_inputs_length: batch_data['q_len_test'],
                                       self.model.decoder_inputs: batch_data['next_q_test'],
                                       self.model.decoder_inputs_length: batch_data['next_q_len_test'],
                                       self.model.input_Y: batch_data['Y_test'],
                                       self.model.lambdas: batch_data['lambda_test'],
                                       self.model.input_mask_pos: self.model.gen_mask(batch_data['q_test'], batch_data['d1_test']),
                                       self.model.input_mask_neg: self.model.gen_mask(batch_data['q_test'], batch_data['d2_test']),
                                       self.model.atten_mask_q: self.model.gen_atten_mask(batch_data['q_test'], self.model.max_q_len),
                                       self.model.atten_mask_pos: self.model.gen_atten_mask(batch_data['d1_test'], self.model.max_d_len),
                                       self.model.atten_mask_neg: self.model.gen_atten_mask(batch_data['d2_test'], self.model.max_d_len),
                                       self.model.atten_mask_q_personal: self.model.gen_atten_mask(batch_data['q_test_personal'], self.model.max_q_len),
                                       self.model.pos_d_mask: self.model.gen_doc_mask(batch_data['d1_test']),
                                       self.model.neg_d_mask: self.model.gen_doc_mask(batch_data['d2_test']),}
                    test_loss_, test_acc_, scores= self.sess.run([self.model.loss, self.model.accuracy, self.model.scores], feed_dict_test)
                    testing_loss += test_loss_
                    testing_acc += test_acc_
                    testing_steps += 1
                return testing_acc/max(testing_steps, 1), testing_loss/max(testing_steps, 1), testing_steps


    def score(self, evaluation, fhand):
        self.dataset.prepare_score_dataset()
        with self.graph.as_default():
            for batch_data in self.dataset.gen_score_batch():
                feed_dict_score = {self.model.input_mu: self.model.mus,
                                 self.model.input_sigma: self.model.sigmas,
                                 self.model.input_q: batch_data['q_test'],
                                 self.model.input_q_personal: batch_data['q_test_personal'],
                                 self.model.input_q_weight: batch_data['q_weight_test'],
                                 self.model.input_pos_d: batch_data['d1_test'],
                                 self.model.input_neg_d: batch_data['d2_test'],
                                 self.model.input_pos_f: batch_data['f1_test'],
                                 self.model.input_neg_f: batch_data['f2_test'],
                                 self.model.encoder_inputs_length: batch_data['q_len_test'],
                                 self.model.decoder_inputs: batch_data['next_q_test'],
                                 self.model.decoder_inputs_length: batch_data['next_q_len_test'],
                                 self.model.input_Y: batch_data['Y_test'],
                                 self.model.lambdas: batch_data['lambda_test'],
                                 self.model.input_mask_pos: self.model.gen_mask(batch_data['q_test'], batch_data['d1_test']),
                                 self.model.input_mask_neg: self.model.gen_mask(batch_data['q_test'], batch_data['d2_test']),
                                 self.model.atten_mask_q: self.model.gen_atten_mask(batch_data['q_test'], self.model.max_q_len),
                                 self.model.atten_mask_pos: self.model.gen_atten_mask(batch_data['d1_test'], self.model.max_d_len),
                                 self.model.atten_mask_neg: self.model.gen_atten_mask(batch_data['d2_test'], self.model.max_d_len),
                                 self.model.atten_mask_q_personal: self.model.gen_atten_mask(batch_data['q_test_personal'], self.model.max_q_len),
                                 self.model.pos_d_mask: self.model.gen_doc_mask(batch_data['d1_test']),
                                 self.model.neg_d_mask: self.model.gen_doc_mask(batch_data['d2_test']),}
                scores = self.sess.run(self.model.scores, feed_dict_score)
                evaluation.write_score(scores, batch_data['lines_test'], fhand) 


    def set_global_weights(self, global_weight):
        """ Assign all the variables in local model with global weights """
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, global_weight):
                variable.load(value, self.sess)


    def set_personal_embed(self):
        """ Assign personal embedding for the user，如果从来没训练过则用global embedding进行初始化，否则用训练之后的embedding进行初始化 """
        personal_embed = np.array(self.personal_embedding)
        if self.user in self.users_word_embedding:
            #print("initialize personal word embedding...")
            for word, embedding in self.users_word_embedding[self.user].items():
                wordid = self.person_vocab[word]
                personal_embed[wordid] = embedding

        with self.graph.as_default():
            self.model.person_embeddings.load(personal_embed, self.sess)


    def save(self, model_dir):
        """
        save the model into model_dir
        """
        with self.graph.as_default():
            self.saver.save(self.sess, model_dir)
        self.logger.info('Model saved in {}.'.format(model_dir))


    def restore(self, model_dir):
        """
        restore the model from the model dir
        """
        with self.graph.as_default():
            self.saver.restore(self.sess, model_dir)
        self.logger.info('Model restored from {}.'.format(model_dir))