from file_manager import FileManager
import tensorflow as tf


class ConfigProject:
    def __init__(self, fm: FileManager, embed_model):
        self.values = { }
        self['embed'] = embed_model(fm['embed'])
        self['count_class'] = None
        self['size_train_batch'] = 1000
        self['size_dev_batch'] = 500
        self['size_test_batch'] = 1000
        self['len(Sent)'] = 20
        self['len(w2v_vector)'] = self['embed'].size_embedding
        self['Float(tf)'] = tf.float32
        self['Int(tf)'] = tf.int32
        self['Bool(tf)'] = tf.bool
        self['output_rnn'] = 128
        self['2_output_rnn'] = 2 * self['output_rnn']
        self['count_rnn_layers'] = 2
        self['do_cell_input'] = 0.6
        self['do_cell_hidden'] = 0.6
        self['do_embeddings'] = 0.2
        self['do_output'] = 0.2
        self['count_epoch'] = 200
        self['l2_reg'] = 0.0001
    
    def __getitem__(self, name):
        return self.values[name]
    
    def __setitem__(self, name, value):
        self.values[name] = value
