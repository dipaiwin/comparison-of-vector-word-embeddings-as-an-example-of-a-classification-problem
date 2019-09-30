import tensorflow as tf


class Values:
    def __init__(self):
        self.values = { }
        self.weights_to_l2_reg = []
    
    def __getitem__(self, name):
        return self.values[name]
    
    def __setitem__(self, name, value):
        self.values[name] = value
    
    def create_placeholder(self, name, shape, dtype, name_tf=None):
        placeholder = tf.placeholder(
            dtype,
            shape,
            name_tf
        )
        self[name] = placeholder
    
    def create_variable(self, name, shape, name_tf=None, add_to_weights=False):
        variable = tf.Variable(
            shape,
            name_tf
        )
        self[name] = variable
        if add_to_weights:
            self.add_weights_name(name)
    
    def add_weights_name(self, name):
        self.weights_to_l2_reg.append(name)
    
    def get_weights(self):
        return [self[name] for name in self.weights_to_l2_reg]
