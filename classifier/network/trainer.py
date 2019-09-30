import tensorflow as tf
from tqdm import trange

from batch_manager import BatchManager as BM
from config_project import ConfigProject
from network.nn_manager import NNManager


class Trainer(NNManager):
    def __init__(self, config: ConfigProject, bs_train: BM, bs_dev: BM, path):
        super().__init__(config, path)
        self.train = bs_train
        self.dev = bs_dev
        self.saver = tf.train.Saver()
        self.best_acc = 0
        self.__init_sess()
    
    def __init_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        self.sess.run(init)
    
    def run(self):
        for epoch in range(self.cfg['count_epoch']):
            print('\nEpoch', epoch + 1)
            for _ in trange(20):
                batch = self.train.get_batch()
                self.sess.run(
                    [self.values['train_step']],
                    {
                        self.values['w2v']: batch['w2v'],
                        self.values['answer']: batch['answer'],
                        self.values['lengths']: batch['lengths'],
                        self.values['is_training']: True
                    }
                )
            self.__test()

    def __test(self):
        batch = self.dev.get_batch()
        loss, acc, net_a = self.sess.run(
            [self.values['answer_loss'], self.values['accuracy'], self.values['net_a']],
            {
                self.values['w2v']: batch['w2v'],
                self.values['answer']: batch['answer'],
                self.values['lengths']: batch['lengths'],
                self.values['is_training']: False
            }
        )
        self.check_save_model(acc)
        self.culc_statistic(batch['answer'], net_a, batch['origin_ids'])
        print('Точность:{0}\tОшибка:{1}\tЛучшее:{2}'.format(acc, loss, self.best_acc))
    
    def __save_model(self):
        sa_pat = self.saver.save(self.sess, self.path)
        print("Model saved in path: %s" % sa_pat)
    
    def check_save_model(self, new_acc):
        if new_acc > self.best_acc:
            self.best_acc = new_acc
            self.__save_model()
