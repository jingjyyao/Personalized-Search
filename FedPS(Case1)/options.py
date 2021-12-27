# 此程序列出训练过程中所需要的所有参数
import argparse

def args_parser():
    parser = argparse.ArgumentParser("peps")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_epoch', default=5, type=int,
                        help='number of global training rounds.')
    parser.add_argument('--num_users', default=10000, type=int,
                        help='total number of users/clients: K')
    parser.add_argument('--fraction', default=0.3, type=float,
                        help='the fraction of clients sampled in each round: C')
    parser.add_argument('--local_epoch', default=10, type=int,
                        help='number of local epochs in each round: E')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='local batch size: B')
    parser.add_argument('--eval_batchsize', default=8, type=int,
                        help='batch size of evaluation')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='learning rate of local training')
    parser.add_argument('--epsilon', default=1e-5, type=float,
                        help='epsilon for AdamOptimizer.')

    # model arguments
    parser.add_argument('--n_bins', default=11, type=int,
                        help='The number of kernels.')
    parser.add_argument('--max_q_len', default=20, type=int,
                        help='Max length of queries.')
    parser.add_argument('--max_d_len', default=50, type=int,
                        help='Max length of documents.')
    parser.add_argument('--embedding_size', default=100, type=int,
                        help='size of the word embedding.')
    parser.add_argument('--vocabulary_size', default=124067, type=int,
                        help='size of the vocabulary.')
    parser.add_argument('--embedding_path', default='/home/jing_yao/learning/personalization/AOL_Data/glove/word2vec.txt',
                        help='path of the word embeddings.')
    parser.add_argument('--vocab_path', default='/home/jing_yao/learning/personalization/AOL_Data/vocab.dict',
                        help='path of the vocabulary.')
    parser.add_argument('--person_vocab_path', default='/home/jing_yao/learning/personalization/AOL_Data/doc_w2v/person_vocab.dict',
                        help='path of the personal vocabulary.')
    parser.add_argument('--users_vocab_path', default='/home/jing_yao/learning/personalization/AOL_Data/doc_w2v/users_vocab.dict',
                        help='path of the vocabulary for each user.')
    parser.add_argument('--person_vocab_size', default=124067, type=int,
                        help='size of the personal vocabulary.')
    parser.add_argument('--with_idf', default=False, type=bool,
                        help='whether take the idf weight into consideration.')
    parser.add_argument('--feature_size', default=98, type=int,
                        help='dimension of the SLTB features.')
    parser.add_argument('--hidden_size', default=100, type=int,
                        help='dimension of hidden state in LSTM.')


    # other arguments
    parser.add_argument('--train', action='store_true',
                        help='whether to train the model.')
    parser.add_argument('--test', action='store_true',
                        help='whether to test the model.')
    parser.add_argument('--score', action='store_true',
                        help='whether to score every document and get the ranking.')
    parser.add_argument('--log_path', help='path of the logging file.')
    parser.add_argument('--model_dir', default='/home/jing_yao/learning/personalized_embedding_2/models/knrm',
                        help='path to store the trained models.')
    parser.add_argument('--restore_path', default='/home/jing_yao/learning/personalized_embedding/models/knrm_0',
                        help='path to restore the model.')
    parser.add_argument('--score_file', default='test_score.txt',
                        help='filename of the test score file')
    parser.add_argument('--in_path', default='/home/jing_yao/learning/personalization/AOL_Data/SampleUsers',
                        help='path of the input. ')
    parser.add_argument('--test_before_aggre', default=False, type=bool,
                        help='whether to test the model before global aggregation.')

    args = parser.parse_args()
    return args