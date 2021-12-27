# 此程序实现了每个客户端本地更新模型
from PEPS import PEPS
from dataset import Dataset
import tensorflow as tf
import logging
import time
import pickle


class GlobalModel:
    def __init__(self, args):
        # logging info
        self.logger = logging.getLogger('peps')

        self.args = args
        self.model_dir = args.model_dir

        # build the model
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.graph.as_default():
            self.model = PEPS(args)
            self.sess.run(tf.global_variables_initializer())
            embed = self.model.load_embedding(args.embedding_path, args.vocab_path, args.vocabulary_size)
            self.model.embeddings.load(embed, self.sess)
            #self.sess.run(tf.assign(self.model.embeddings, embed))
        
            # save info
            self.saver = tf.train.Saver()

    def get_global_weights(self):
        with self.graph.as_default():
            global_weight = self.sess.run(tf.trainable_variables())
        return global_weight

    def set_global_weights(self, global_weight):
        """ Assign all the variables in local model with global weights """
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, global_weight):
                variable.load(value, self.sess)


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