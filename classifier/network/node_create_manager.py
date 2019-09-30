import tensorflow as tf
from network.values_manager import Values
from config_project import ConfigProject


class NodeCreateManager:
    def __init__(self, config: ConfigProject):
        self.values = Values()
        self.cfg = config
    
    def __getitem__(self, name_node):
        return self.values[name_node]
    
    def __setitem__(self, name_node, value):
        self.values[name_node] = value
    
    def create_nodes(self):
        self.values.create_placeholder(
            'w2v',
            [None, self.cfg['len(Sent)'], self.cfg['len(w2v_vector)']],
            self.cfg['Float(tf)']
        )
        self.values.create_placeholder(
            'answer',
            [None],
            self.cfg['Int(tf)']
        )
        self.values.create_placeholder(
            'lengths',
            [None],
            self.cfg['Int(tf)']
        )
        self.values.create_placeholder(
            'is_training',
            [],
            self.cfg['Bool(tf)']
        )
        self.values.create_variable(
            'weights',
            tf.truncated_normal(
                name='weights',
                shape=[
                    self.cfg['2_output_rnn'],
                    self.cfg['count_class']
                ],
                stddev=0.1
            ),
            add_to_weights=True
        )
        self.values.create_variable(
            'biases',
            tf.random_normal(
                [self.cfg['count_class']]
            ),
            add_to_weights=True
        )
        return self.values
