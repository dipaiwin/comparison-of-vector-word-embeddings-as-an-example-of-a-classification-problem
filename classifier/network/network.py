import tensorflow as tf
from network.values_manager import Values
from config_project import ConfigProject
from network.rnn_manager import RnnManager


class ManagerNetwork:
    def __init__(self, values: Values, config: ConfigProject):
        self.values = values
        self.cfg = config
    
    def create_network(self):
        batch_size = tf.shape(self.values['w2v'])[0]
        self.values['w2v'] = tf.multiply(
            self.values['w2v'],
            tf.layers.dropout(
                tf.ones([batch_size, 1, self.cfg['len(w2v_vector)']]),
                self.cfg['do_embeddings'],
                training=self.values['is_training']
            )
        )
        
        self.values['post_rnn'] = RnnManager(self.cfg, self.values).run_rnn(self.values['w2v'])
        self.values['post_rnn'] = self.values['post_rnn'][:, 0, :]
        self.values['post_rnn'] = self.dropout(self.values['post_rnn'])
        self.values['res'] = tf.einsum('ij,jk->ik', self.values['post_rnn'], self.values['weights']) + self.values[
            'biases']
        self.values['activation'] = tf.nn.softmax(self.values['res'], axis=1)
        self.values['net_a'] = tf.argmax(self.values['activation'], axis=1)
        self.values['answer_loss'] = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.values['res'],
                labels=self.values['answer']
            )
        )
        # l2 = 0.0001
        self.values['l2_loss'] = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.cfg['l2_reg']),
            self.values.get_weights()
        )
        self.values['loss'] = self.values['answer_loss'] + self.values["l2_loss"]
        self.values['train_step'] = tf.train.AdamOptimizer().minimize(self.values['loss'])
        self.values['correct_pred'] = tf.equal(self.values['net_a'], tf.cast(self.values['answer'], 'int64'))
        self.values['accuracy'] = tf.reduce_mean(tf.cast(self.values['correct_pred'], 'float'))
    
    def dropout(self, matrix, scale=None):
        return tf.layers.dropout(
            matrix,
            scale if scale is not None else self.cfg['do_output'],
            training=self.values['is_training']
        )
