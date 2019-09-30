import tensorflow as tf

from batch_manager import BatchManager as bm
from config_project import ConfigProject
from network.nn_manager import NNManager


class Tester(NNManager):
    def __init__(self, config: ConfigProject, bs_test: bm, path):
        super().__init__(config, path)
        self.__test = bs_test
        self.__load_sess()
    
    def __load_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, self.path)
    
    def run(self):
        test_epoch = 10
        total_acc = 0
        for i in range(test_epoch):
            batch = self.__test.get_batch()
            loss, acc, net_a = self.sess.run(
                [self.values['answer_loss'], self.values['accuracy'], self.values['net_a']],
                {
                    self.values['w2v']: batch['w2v'],
                    self.values['answer']: batch['answer'],
                    self.values['lengths']: batch['lengths'],
                    self.values['is_training']: False
                }
            )
            self.culc_statistic(batch['answer'], net_a, batch['origin_ids'])
            total_acc += acc
            print('Точность:{0}\tОшибка:{1}'.format(acc, loss))
        final_acc = total_acc / test_epoch
        print('Общая точность:{0}'.format(final_acc))
        return final_acc
